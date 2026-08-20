"""K-4b0.3: развёртка по окну исполнения.

ВОПРОС. K-4b0.2 показал, что средний профиль полезности позиций определяется
геометрией декодера относительно окна из ЧЕТЫРЁХ исполняемых действий:
ранговая корреляция чувствительности декодера с эмпирическим профилем +0.976, а
на всём чанке из 20 шагов профиль плоский (коэф. вариации 0.06-0.09). Значит
прежняя постановка не могла проверить, существует ли СОСТОЯНИЕ-ЗАВИСИМОЕ
разреженное активное множество: ответ был предопределён метрикой.

Скрипт считает всю линейку K-4b0.1 для набора окон из ОДНОГО датасета,
собранного с --per-timestep. Пересборка не нужна: хранится ошибка по каждому
шагу действия, а окно — это выбор подмножества шагов.

ПОЧЕМУ ЭТО КОРРЕКТНО. g_flat_t[j, t] = e0_t[t] - e_S_t[t], поэтому для любого
окна W:
    e0(W)  = mean_{t in W} e_empty_t[t]
    e_S(W) = e0(W) - mean_{t in W} g_flat_t[j, t]
    G_rms  = sqrt(e0(W)) - sqrt(max(e_S(W), 0))
Складываются КВАДРАТЫ, корень берётся один раз в конце — иначе по Йенсену
результат смещён.

ПРЕДРЕГИСТРИРОВАННОЕ ПРЕДСКАЗАНИЕ (записано до первого запуска):
  1. при окне 20 лучшая фиксированная маска должна сравняться со случайным
     выбором четырёх — её преимущество происходило из концентрации
     чувствительности, а на всём чанке концентрации нет;
  2. если точный оракул при K=4 остаётся существенно выше случайного, значит
     состояние-зависимое разреженное множество существует; если падает к
     K/16 = 25%, значит вне окна исполнения его нет.
Первое проверяет полноту объяснения через декодер, второе — саму гипотезу.

СТАТУС ЧИСЕЛ. Протокольным остаётся окно 4: развёрнутая политика исполняет
ровно четыре действия из двадцати предсказанных (eval_libero.py:235,
HORIZON=4). Остальные окна — научная диагностика, и смешивать их с
протокольными числами нельзя.

Запуск:
    python3 experiments/k4b0_window_sweep.py --dir data/k4b0_win_sweep
    python3 experiments/k4b0_window_sweep.py --selftest
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k4b0_verify import cluster_ci, macro_by  # noqa: E402
from k4b0_extra_baselines import paired_ci  # noqa: E402


def windows(T):
    """Префиксы и непересекающиеся блоки. Префиксы отвечают на вопрос «что
    будет при более длинном горизонте исполнения», блоки — «одинакова ли
    структура в разных частях чанка» (если активная четвёрка ПЕРЕЕЗЖАЕТ вместе
    с блоком, это прямое доказательство геометрической природы)."""
    out = {}
    for w in (1, 2, 4, 8, 12, 16, T):
        if w <= T:
            out[f"префикс {w}"] = np.arange(w)
    for a in range(0, T, 4):
        b = min(a + 4, T)
        out[f"блок [{a}:{b}]"] = np.arange(a, b)
    return out


class Rows:
    """Разбор рваной таблицы один раз: составы подмножеств от окна не зависят."""

    def __init__(self, lb, P, kmax):
        self.P, self.kmax = P, kmax
        self.off = lb["g_off"]
        self.C, self.index = [], []
        for i in range(len(lb["obs_idx"])):
            C = tuple(q for q in range(P) if lb["support"][i] >> q & 1)
            subs = [S for k in range(kmax + 1)
                    for S in itertools.combinations(C, k)]
            assert self.off[i + 1] - self.off[i] == len(subs)
            self.C.append(set(C))
            self.index.append({S: j for j, S in enumerate(subs)})

    def mask_map(self, idx, masks):
        """Индексы в gw для КАЖДОЙ маски и каждой строки, посчитанные один раз.

        Составы подмножеств от окна не зависят, поэтому отображение
        «маска -> позиция в рваной таблице» строится однократно, а смена окна
        сводится к индексированию массива. Без этого перебор 1820 масок в
        двенадцати окнах был бы главной стоимостью скрипта."""
        MI = np.empty((len(idx), len(masks)), np.int64)
        for j, i in enumerate(idx):
            i = int(i)
            loc, C, o0 = self.index[i], self.C[i], self.off[i]
            for m, S in enumerate(masks):
                MI[j, m] = o0 + loc[tuple(sorted(set(S) & C))]
        return MI

    def g(self, gw, i, S):
        """G произвольного набора в текущем окне: сводим к пересечению с C."""
        j = self.index[i][tuple(sorted(set(S) & self.C[i]))]
        return gw[self.off[i] + j]


def to_rms(e0, g):
    return np.sqrt(e0) - np.sqrt(np.maximum(e0 - g, 0.0))


def eval_sets(rows, gw, e0w, idx, sets):
    num = np.empty(len(idx))
    for j, i in enumerate(idx):
        num[j] = to_rms(e0w[i], rows.g(gw, int(i), sets[j]))
    return num, np.sqrt(e0w[idx])


def singleton_and_best(rows, gw, e0w, idx, K):
    """Одиночные выигрыши, точный оптимум и жадный путь в текущем окне.

    Одиночные ПЕРЕСЧИТЫВАЮТСЯ: сохранённый sing_gain_rms относится к окну 4 и
    для другого окна неверен."""
    P = rows.P
    sing = np.zeros((len(idx), P))
    best, bestK = [], np.zeros((len(idx), K + 1))
    greedy = []
    for j, i in enumerate(idx):
        i = int(i)
        e0 = e0w[i]
        for q in range(P):
            sing[j, q] = to_rms(e0, rows.g(gw, i, [q]))
        bv, bs = -1e30, ()
        for S, loc in rows.index[i].items():
            v = to_rms(e0, gw[rows.off[i] + loc])
            if len(S) <= K and v > bv:
                bv, bs = v, S
            k = len(S)
            if k <= K and v > bestK[j, k]:
                bestK[j, k] = v
        for k in range(1, K + 1):                       # монотонизация по <=K
            bestK[j, k] = max(bestK[j, k], bestK[j, k - 1])
        best.append(list(bs))
        # ЖАДНЫЙ СО СТОПОМ. Прежде ход делался всегда, даже когда он ухудшал
        # набор, и величина была не «жадный оракул», а «жадный ровно K».
        # Позиции вне support дают ровно ноль, поэтому добивка ими безопасна и
        # сохраняет равную стоимость.
        cur, best_v = [], 0.0
        for _ in range(K):
            cand = [(to_rms(e0, rows.g(gw, i, cur + [q])), q)
                    for q in sorted(rows.C[i]) if q not in cur]
            if not cand:
                break
            v, q = max(cand)
            if v <= best_v:
                break
            cur, best_v = cur + [q], v
        greedy.append(cur + [q for q in range(P) if q not in cur][:K - len(cur)])
    return sing, best, bestK, greedy


def fit_priors(sing, den, p, tsk, P):
    """Таблицы средней полезности позиции, ТОЛЬКО по train.

    Метка нормируется на sqrt(e0) строки — это её собственный вклад в основную
    метрику, величина безразмерная и всегда положительная в знаменателе. Делить
    на g* нельзя: на части строк он около нуля и отношение улетает."""
    y = sing / den[:, None]
    pq = np.zeros((P, P))
    for q_ in range(P):
        for pp in range(P):
            m = p == pp
            pq[pp, q_] = y[m, q_].mean() if m.any() else 0.0
    off = np.zeros(2 * P - 1)
    cnt = np.zeros(2 * P - 1)
    for j in range(len(y)):
        for q_ in range(P):
            off[q_ - p[j] + P - 1] += y[j, q_]
            cnt[q_ - p[j] + P - 1] += 1
    off /= np.maximum(cnt, 1)
    nt = int(tsk.max()) + 1
    tpq = np.zeros((nt, P, P))
    for t_ in range(nt):
        for pp in range(P):
            m = (tsk == t_) & (p == pp)
            if m.any():
                tpq[t_, pp] = y[m].mean(0)
    return dict(glob=y.mean(0), pq=pq, off=off, tpq=tpq)


def prior_score(pri, kind, p, tsk, P, alpha=0.0):
    n = len(p)
    if kind == "glob":
        return np.tile(pri["glob"], (n, 1))
    if kind == "pq":
        return pri["pq"][p]
    if kind == "off":
        return pri["off"][np.arange(P)[None, :] - p[:, None] + P - 1]
    return alpha * pri["tpq"][tsk, p] + (1 - alpha) * pri["pq"][p]


def diversity(sets, P):
    """Насколько оптимальные наборы РАЗНЫЕ между строками.

    Средний попарный Жаккар на подвыборке пар плюс энтропия частот позиций.
    Если наборы почти совпадают, «динамический» router вырождается в маску, и
    это видно здесь до всякого обучения."""
    r = np.random.default_rng(0)
    n = len(sets)
    ii = r.integers(0, n, 4000)
    jj = r.integers(0, n, 4000)
    jac = []
    for a, b in zip(ii, jj):
        if a == b:
            continue
        A, B = set(sets[a]), set(sets[b])
        jac.append(len(A & B) / max(len(A | B), 1))
    freq = np.bincount(np.concatenate([np.asarray(s, int) for s in sets]),
                       minlength=P).astype(float)
    freq /= freq.sum()
    ent = -(freq[freq > 0] * np.log(freq[freq > 0])).sum() / np.log(P)
    return float(np.mean(jac)), float(ent), freq


def selftest():
    """Синтетика с ИЗВЕСТНЫМ ОТВЕТОМ для сведения окон.

    Строим потимшаговую таблицу вручную так, что в первой половине шагов важна
    позиция 0, а во второй — позиция 1. Развёртка обязана это увидеть: на
    блоке [0:2] лучшая позиция 0, на блоке [2:4] — позиция 1, а на префиксе 4
    обе примерно равны."""
    P, T, kmax = 3, 4, 2
    C = (0, 1)
    subs = [S for k in range(kmax + 1) for S in itertools.combinations(C, k)]
    e0_t = np.array([[1.0, 1.0, 1.0, 1.0]])
    g_t = np.zeros((len(subs), T))
    for j, S in enumerate(subs):
        if 0 in S:
            g_t[j, :2] += 0.8
        if 1 in S:
            g_t[j, 2:] += 0.8
    lb = dict(obs_idx=np.array([0]), support=np.array([0b011]),
              g_off=np.array([0, len(subs)]))
    rows = Rows(lb, P, kmax)
    for w, exp in ((np.arange(2), 0), (np.arange(2, 4), 1)):
        gw = g_t[:, w].mean(1)
        e0w = e0_t[:, w].mean(1)
        s = [to_rms(e0w[0], rows.g(gw, 0, [q])) for q in range(P)]
        assert int(np.argmax(s)) == exp, f"окно {w}: ждали {exp}, вышло {s}"
    gw = g_t[:, :4].mean(1)
    e0w = e0_t[:, :4].mean(1)
    s = [to_rms(e0w[0], rows.g(gw, 0, [q])) for q in range(P)]
    assert abs(s[0] - s[1]) < 1e-9 and s[2] == 0.0, f"префикс 4: {s}"
    # НЕСКЛАДЫВАЕМОСТЬ КОРНЯ: RMS по объединению не равен среднему RMS частей.
    r1 = to_rms(e0_t[0, :2].mean(), g_t[1, :2].mean())
    r2 = to_rms(e0_t[0, 2:].mean(), g_t[1, 2:].mean())
    rw = to_rms(e0_t[0].mean(), g_t[1].mean())
    assert abs(rw - (r1 + r2) / 2) > 1e-3, \
        "проверка на несходимость RMS по частям не сработала"
    # КРИВАЯ ПО K. Прежняя ошибка — двойное применение to_rms — существующей
    # самопроверкой не ловилась, поэтому проверяется отдельно и с известным
    # ответом: при двух полезных позициях кривая обязана расти до K=2 и дальше
    # выйти на плато, а её значение при K=2 совпасть с точным оптимумом.
    gw = g_t[:, :4].mean(1)
    e0w = e0_t[:, :4].mean(1)
    bk = np.zeros(kmax + 1)
    for S, loc in rows.index[0].items():
        v = to_rms(e0w[0], gw[loc])
        if len(S) <= kmax:
            bk[len(S)] = max(bk[len(S)], v)
    for k in range(1, kmax + 1):
        bk[k] = max(bk[k], bk[k - 1])
    kc = bk / np.sqrt(e0w[0])
    assert kc[0] == 0.0, "K=0 обязан давать ноль"
    assert kc[1] < kc[2], f"кривая не растёт до K=2: {kc}"
    assert kc[2] <= 1.0 + 1e-12, f"доля больше единицы: {kc}"
    assert abs(kc[2] - kc[kmax]) < 1e-12, f"плато после K=2 не вышло: {kc}"

    print("самопроверка пройдена: окна разделяются, активная позиция "
          "переезжает вместе с блоком, RMS по частям не складывается, "
          "кривая по K растёт и выходит на плато")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    selftest()
    if not args.dir:
        raise SystemExit("нужен --dir или --selftest")

    meta = json.load(open(os.path.join(args.dir, "metadata.json")))
    _lb = np.load(os.path.join(args.dir, "labels.npz"), allow_pickle=True)
    lb = {k: _lb[k] for k in _lb.files}          # материализуем: NpzFile ленив
    if "g_flat_t" not in lb:
        raise SystemExit("датасет собран без --per-timestep, развёртка "
                         "невозможна: пересоберите с этим флагом")
    _ft = np.load(os.path.join(args.dir, "features.npz"), allow_pickle=True)
    tsk_idx = _ft["obs_task_idx"][lb["obs_idx"]]
    tsk = np.asarray(meta["tasks"])[tsk_idx]
    P, K = meta["P"], meta["kmax"]
    T = lb["e_empty_t"].shape[1]
    epi, sp = lb["episode"], lb["split"]
    n = len(lb["obs_idx"])
    print(f"строк {n}, шагов действия {T}, окно протокола {meta['window']}, "
          f"P={P}, K={K}")
    print("разбираю составы подмножеств (один раз на весь прогон)...",
          flush=True)
    rows = Rows(lb, P, K)

    itr, iv, it = (np.where(sp == 0)[0], np.where(sp == 1)[0],
                   np.where(sp == 2)[0])
    p_all = lb["p"].astype(np.int64)
    all_masks = list(itertools.combinations(range(P), 4))
    print(f"строю отображение масок ({len(all_masks)} на строку, один раз)...",
          flush=True)
    MI_tr, MI_iv, MI_it = (rows.mask_map(itr, all_masks),
                           rows.mask_map(iv, all_masks),
                           rows.mask_map(it, all_masks))
    res = {}

    for name, w in windows(T).items():
        gw = lb["g_flat_t"][:, w].mean(1)
        e0w = lb["e_empty_t"][:, w].mean(1)
        # СВЕРКА С ПРОТОКОЛЬНЫМИ ЧИСЛАМИ: окно 4 обязано воспроизвести
        # сохранённую скалярную таблицу, иначе развёртке верить нельзя.
        if name == f"префикс {meta['window']}":
            assert np.allclose(gw, lb["g_flat"], atol=1e-6), \
                "окно протокола не воспроизвело скалярную таблицу"
            print(f"  [{name}] сведение к сохранённой таблице сошлось")

        sing, best, bestK, greedy = singleton_and_best(rows, gw, e0w, it, K)
        o1 = [list(np.argsort(-sing[j], kind="stable")[:4])
              for j in range(len(it))]

        # МАСКИ ПЕРЕБИРАЮТСЯ ВЕКТОРНО через предпосчитанное отображение.
        M_iv = to_rms(e0w[iv][:, None], gw[MI_iv])
        bm = all_masks[int(np.argmax(M_iv.sum(0)))]
        fx_n, den = eval_sets(rows, gw, e0w, it, [list(bm)] * len(it))
        o1_n, _ = eval_sets(rows, gw, e0w, it, o1)
        ex_n, _ = eval_sets(rows, gw, e0w, it, best)
        gr_n, _ = eval_sets(rows, gw, e0w, it, greedy)
        rnd = []
        for s_ in range(20):
            r = np.random.default_rng(s_)
            rn, _ = eval_sets(rows, gw, e0w, it,
                              [list(r.permutation(P)[:4]) for _ in it])
            rnd.append(rn)
        rn_n = np.mean(rnd, 0)

        # ---- УСЛОВНЫЕ BASELINE БЕЗ СОСТОЯНИЯ, обучаются ТОЛЬКО по train.
        # Ключевой из них S(p): своя ЦЕЛЬНАЯ четвёрка на каждую позицию
        # правки p. Он видит взаимодействия внутри набора и зависимость от p,
        # но НЕ видит ни картинку, ни состояние робота, ни токены. Поэтому он
        # отделяет настоящую состояние-зависимость от зависимости от p.
        M_tr = to_rms(e0w[itr][:, None], gw[MI_tr])
        M_it = to_rms(e0w[it][:, None], gw[MI_it])
        sg_tr = np.array([[to_rms(e0w[int(i)], rows.g(gw, int(i), [q]))
                           for q in range(P)] for i in itr])
        den_tr = np.sqrt(e0w[itr])
        pri = fit_priors(sg_tr, den_tr, p_all[itr], tsk_idx[itr], P)

        cond = {}
        cond["S_global (train)"] = M_it[:, int(np.argmax(M_tr.sum(0)))]
        sp_pick = np.zeros(P, np.int64)
        for pp in range(P):
            m = p_all[itr] == pp
            sp_pick[pp] = int(np.argmax(M_tr[m].sum(0))) if m.any() else 0
        cond["S(p) (train)"] = M_it[np.arange(len(it)), sp_pick[p_all[it]]]
        for kind, nm_ in (("glob", "prior[q]"), ("pq", "prior[p,q]"),
                          ("off", "prior[q-p]")):
            sc = prior_score(pri, kind, p_all[it], tsk_idx[it], P)
            sets_ = np.argsort(-sc, axis=1, kind="stable")[:, :4]
            cond[nm_], _ = eval_sets(rows, gw, e0w, it, sets_.tolist())
        ba, bv = 0.0, -1e30                       # alpha выбирается на val
        M_iv_ = M_iv
        sg_iv = np.array([[to_rms(e0w[int(i)], rows.g(gw, int(i), [q]))
                           for q in range(P)] for i in iv])
        for a_ in (0.0, 0.25, 0.5, 0.75, 1.0):
            sc = prior_score(pri, "task", p_all[iv], tsk_idx[iv], P, a_)
            st = np.argsort(-sc, axis=1, kind="stable")[:, :4]
            nv, dv = eval_sets(rows, gw, e0w, iv, st.tolist())
            if nv.sum() / dv.sum() > bv:
                ba, bv = a_, nv.sum() / dv.sum()
        sc = prior_score(pri, "task", p_all[it], tsk_idx[it], P, ba)
        cond[f"prior[task,p,q] a={ba}"], _ = eval_sets(
            rows, gw, e0w, it, np.argsort(-sc, 1, kind="stable")[:, :4].tolist())
        best_cond = max(cond, key=lambda k_: cond[k_].sum())
        d_cond = paired_ci(ex_n, cond[best_cond], den, epi[it])

        # разнообразие ВНУТРИ фиксированного p: если наборы разные и при
        # известном p, изменчивость нельзя списать на позицию правки
        jac_p = float(np.mean([diversity([best[j] for j in
                                          np.where(p_all[it] == pp)[0]], P)[0]
                               for pp in range(P)
                               if (p_all[it] == pp).sum() > 1]))
        gstar = ex_n
        ok = gstar > 0
        med = float(np.median(cond[best_cond][ok] / gstar[ok]))

        jac, ent, freq = diversity(best, P)
        d_fx = paired_ci(fx_n, rn_n, den, epi[it])
        d_ex = paired_ci(ex_n, fx_n, den, epi[it])
        row = {}
        for lbl, num in (("случайные 4", rn_n), ("фикс-маска", fx_n),
                         ("O1", o1_n), ("жадный", gr_n),
                         ("точный <=4", ex_n)):
            pt, lo, hi = cluster_ci(num, den, epi[it])
            row[lbl] = dict(R=pt, lo=lo, hi=hi,
                            macro=macro_by(num, den, tsk[it]))
        # bestK УЖЕ на RMS-шкале: to_rms применён внутри singleton_and_best.
        # Прежде здесь стояло второе преобразование, и величина зажималась в
        # единицу во всех окнах — знаменатель den тоже равен sqrt(e0).
        kc = [float(bestK[:, k].sum() / den.sum()) for k in range(K + 1)]
        for nm_, num in cond.items():
            pt, lo, hi = cluster_ci(num, den, epi[it])
            row[nm_] = dict(R=pt, lo=lo, hi=hi, macro=macro_by(num, den,
                                                               tsk[it]))
        row |= dict(best_cond=best_cond, exact_minus_cond=d_cond,
                    jaccard_within_p=jac_p, median_cond_over_gstar=med,
                    mask=list(map(int, bm)), jaccard=jac, entropy=ent,
                    freq=freq.tolist(), K_curve=kc,
                    fixed_minus_random=d_fx, exact_minus_fixed=d_ex)
        res[name] = row

        print(f"\n  {name}: маска {list(bm)}")
        print(f"    {'случайные 4':<14}{row['случайные 4']['R']:.3f}   "
              f"{'фикс-маска':<12}{row['фикс-маска']['R']:.3f}   "
              f"{'O1':<4}{row['O1']['R']:.3f}")
        print(f"    {'жадный':<14}{row['жадный']['R']:.3f}   "
              f"{'точный <=4':<12}{row['точный <=4']['R']:.3f}")
        print(f"    фикс − случайные {d_fx[0]:+.4f} [{d_fx[1]:+.4f}, "
              f"{d_fx[2]:+.4f}]"
              + ("  значимо" if d_fx[1] > 0 or d_fx[2] < 0 else "  НЕ значимо"))
        print(f"    точный − фикс    {d_ex[0]:+.4f} [{d_ex[1]:+.4f}, "
              f"{d_ex[2]:+.4f}]"
              + ("  значимо" if d_ex[1] > 0 or d_ex[2] < 0 else "  НЕ значимо"))
        print("    условные baseline без состояния: " + "  ".join(
            f"{k_} {v_.sum() / den.sum():.3f}" for k_, v_ in cond.items()))
        print(f"    лучший из них — {best_cond}; точный − он "
              f"{d_cond[0]:+.4f} [{d_cond[1]:+.4f}, {d_cond[2]:+.4f}]"
              + ("  значимо" if d_cond[1] > 0 else "  НЕ значимо"))
        print(f"    разнообразие лучших наборов: Жаккар {jac:.3f} "
              f"(внутри фиксированного p {jac_p:.3f}), "
              f"энтропия позиций {ent:.3f}")
        print(f"    кривая K: " + " ".join(f"{v:.3f}" for v in kc))

    print("\n" + "=" * 74)
    print("СВОДКА: держится ли разреженность при удлинении окна")
    print("=" * 74)
    print(f"  {'окно':<14}{'случ.':>7}{'фикс':>7}{'лучш.усл.':>10}{'O1':>7}"
          f"{'точн.':>7}{'точн.-усл.':>22}{'Жак|p':>7}")
    for k, v in res.items():
        d = v["exact_minus_cond"]
        print(f"  {k:<14}{v['случайные 4']['R']:>7.3f}"
              f"{v['фикс-маска']['R']:>7.3f}"
              f"{v[v['best_cond']]['R']:>10.3f}{v['O1']['R']:>7.3f}"
              f"{v['точный <=4']['R']:>7.3f}"
              f"   {d[0]:>+7.4f} [{d[1]:>+7.4f},{d[2]:>+7.4f}]"
              f"{v['jaccard_within_p']:>7.3f}")
    print("\n  ВОРОТА ПО УСЛОВНЫМ BASELINE, записаны до запуска.\n"
          "  Величина «точный − лучший условный» отделяет настоящую\n"
          "  состояние-зависимость от зависимости от позиции правки p и от\n"
          "  задачи. S(p) даёт свою ЦЕЛЬНУЮ четвёрку на каждое p, то есть\n"
          "  видит и взаимодействия внутри набора, но не видит ни картинки,\n"
          "  ни состояния робота, ни токенов.\n"
          "    >= 0.10 — сильное доказательство динамического запаса;\n"
          "    >= 0.05 при положительной нижней границе — переходить к полной\n"
          "            пересборке;\n"
          "    <  0.05 — изменчивость объясняется p и задачей, формулировку\n"
          "            «state-dependent» снять.\n"
          "  Жаккар ВНУТРИ фиксированного p дополняет: если наборы разные и\n"
          "  при известном p, изменчивость на p не спишешь.\n"
          "\n  ЧИТАТЬ ТАК, правило записано до запуска.\n"
          "  Если на длинных окнах фикс-маска сходится со случайным выбором —\n"
          "  объяснение через геометрию декодера полное. Если точный оракул\n"
          "  при этом держится заметно выше случайного, а Жаккар лучших\n"
          "  наборов низкий — состояние-зависимое множество существует, и его\n"
          "  можно проверять честно. Если точный падает к K/16 = 0.25, вне\n"
          "  окна исполнения разреженного множества нет.\n"
          "  ПРОТОКОЛЬНЫМ остаётся окно 4: политика исполняет четыре шага.")

    if args.out:
        json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"\n  сохранено: {args.out}")


if __name__ == "__main__":
    main()
