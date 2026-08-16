"""K-4a4: прямой тест субмодулярности функции группового ремонта.

ЗАЧЕМ. В фазе A измерено, что СРЕДНЕЕ Delta для трёх способов выбора
отрицательно. Это утверждение про три конкретных набора, а не про функцию.
Субмодулярность — условие на ВСЕ приращения, и его надо проверять прямо.

ЧТО СЧИТАЕМ. В позициях набора S устаревшая латента заменяется опорной:

    e(S) = MSE(Dec(h_stale->S), a_ref),   G(S) = e(пусто) - e(S).

Всё на КВАДРАТЕ ошибки: корень вогнут и сам порождает мнимую
супераддитивность (см. LESSONS.md).

    Omega(A; q, r) = G(A+q) + G(A+r) - G(A) - G(A+q+r) >= 0   (убывающая отдача)
    M(A, q)        = G(A+q) - G(A)                            (монотонность)

ВЫРОЖДЕННОСТЬ. Если позиция не изменилась, её латента совпадает с опорной
ПОБИТОВО, поэтому G(A+q) = G(A) точно и Omega = 0 тождественно. Содержательны
только тройки, где ОБЕ позиции q, r лежат в changed support. Все доли считаются
по содержательным; деление на все тройки занижает результат примерно в десять
раз. То же для монотонности (содержательна, если изменена q) и для СЛОЁВ по
силе позиции: маска top-4 обязана пересекаться с маской содержательных, иначе
при support меньше четырёх в слой попадают позиции с нулевым выигрышем и
искусственно занижают долю нарушений примерно на четверть.

ПРО ГАРАНТИЮ ЖАДНОГО ОТБОРА. Величина gamma = [m(q|A)+m(r|A)] / m(qr|A)
приводится только как ОПИСАТЕЛЬНАЯ локальная парная статистика. Это НЕ
submodularity ratio из теорем Das & Kempe: тот определяется как минимум по
допустимым парам множеств, а не потройно, и требует монотонности функции,
которая здесь нарушается. Подстановка процентиля в 1 - exp(-gamma) гарантией
не является и не вычисляется.

ТОЧНОСТЬ. Пол измеряется НЕПОСРЕДСТВЕННО ДЛЯ Omega (разность четырёх величин),
а не для одиночных выигрышей: у Omega он заведомо выше. Сравниваются полные
таблицы G во float32 и float64 на отдельном вмешательстве.

ОБЪЁМ. Таблица G для всех вмешательств — 1536 x 2517 x 4 байта = 15.5 МБ,
поэтому сохраняется ЦЕЛИКОМ вместе с масками, эпизодами, позициями и порогами.
Режим --from-npz пересчитывает всю статистику без модели и без кодека.

Запуск:
    python3 experiments/k4a4_submodularity_audit.py \
        --ckpt ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO
    python3 experiments/k4a4_submodularity_audit.py --from-npz logs/k4a4.npz
"""

import argparse
import itertools
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TAU_SWEEP = (1e-4, 1e-3, 1e-2)


def build_sets(P: int, kmax: int = 4):
    """Все наборы размера 0..kmax. Пустой ИДЁТ ПЕРВЫМ и имеет G = 0."""
    sets = [()] + [S for k in range(1, kmax + 1)
                   for S in itertools.combinations(range(P), k)]
    return sets, {S: i for i, S in enumerate(sets)}


def build_triples(P: int, idx: dict, max_a: int = 2):
    """Тройки (A, q, r): |A| <= max_a, q < r, обе вне A."""
    iA, iAq, iAr, iAqr, um, qm, na = [], [], [], [], [], [], []
    for ka in range(max_a + 1):
        for A in itertools.combinations(range(P), ka):
            rest = [x for x in range(P) if x not in A]
            for q, r in itertools.combinations(rest, 2):
                iA.append(idx[A])
                iAq.append(idx[tuple(sorted(A + (q,)))])
                iAr.append(idx[tuple(sorted(A + (r,)))])
                iAqr.append(idx[tuple(sorted(A + (q, r)))])
                m = 0
                for x in A + (q, r):
                    m |= 1 << x
                um.append(m)
                qm.append((1 << q) | (1 << r))
                na.append(ka)
    return tuple(np.asarray(v) for v in (iA, iAq, iAr, iAqr, um, qm, na))


def build_mono(P: int, idx: dict, max_a: int = 3):
    """Пары (A, q): |A| <= max_a, q вне A."""
    iA, iAq, um, qm, na = [], [], [], [], []
    for ka in range(max_a + 1):
        for A in itertools.combinations(range(P), ka):
            for q in range(P):
                if q in A:
                    continue
                iA.append(idx[A])
                iAq.append(idx[tuple(sorted(A + (q,)))])
                m = 0
                for x in A + (q,):
                    m |= 1 << x
                um.append(m)
                qm.append(1 << q)
                na.append(ka)
    return tuple(np.asarray(v) for v in (iA, iAq, um, qm, na))


def selftest(P: int = 8) -> None:
    """СТРОГАЯ проверка счётчика на функциях с АНАЛИТИЧЕСКИ известным ответом.

    Основа модулярная (никакой случайной функции покрытия — результат должен
    быть детерминированным):

      1. чисто модулярная               -> Omega = 0 всюду, нарушений 0;
      2. модулярная минус парный штраф  -> Omega = c_qr >= 0, нарушений 0
         (субмодулярна по построению);
      3. модулярная плюс парный БОНУС b на паре {a,b} -> Omega = -b ровно на
         тройках, где {q,r} = {a,b} и A не содержит ни a, ни b. Их число
         равно 1 + (P-2) + C(P-2,2) — проверяется ТОЧНОЕ совпадение."""
    sets, idx = build_sets(P)
    iA, iAq, iAr, iAqr, _, _, _ = build_triples(P, idx)
    w = np.arange(1, P + 1, dtype=float)          # детерминированные веса
    g_mod = np.array([w[list(S)].sum() if S else 0.0 for S in sets])

    def pairs_in(S):
        return list(itertools.combinations(sorted(S), 2))

    c = 0.3
    g_sub = g_mod - np.array([c * len(pairs_in(S)) for S in sets])
    a, b, bonus = 2, 5, 3.0
    g_sup = g_mod + np.array([bonus if {a, b} <= set(S) else 0.0 for S in sets])

    exp_sup = 1 + (P - 2) + (P - 2) * (P - 3) // 2
    print("САМОПРОВЕРКА счётчика (основа модулярная, ответ аналитический):")
    for nm, g, exp in (("модулярная", g_mod, 0),
                       ("модулярная - парный штраф", g_sub, 0),
                       (f"модулярная + бонус на паре {a},{b}", g_sup, exp_sup)):
        om = g[iAq] + g[iAr] - g[iA] - g[iAqr]
        n = int((om < -1e-9).sum())
        ok = "ok" if n == exp else "ПРОВАЛ"
        print(f"  {nm:>34}: нарушений {n:>4}, ожидалось {exp:>4}  {ok}")
        if n != exp:
            raise SystemExit(f"самопроверка провалена на «{nm}»")
    print("  счётчик даёт ТОЧНО предсказанное число нарушений\n")


def cluster_ci(num, den, epi, n_boot: int = 2000, seed: int = 0):
    """Отношение сумм с кластерным бутстрапом по эпизодам."""
    rng = np.random.default_rng(seed)
    eps = np.unique(epi)
    ix = {e: np.where(epi == e)[0] for e in eps}
    pt = num.sum() / max(den.sum(), 1e-30)
    out = []
    for _ in range(n_boot):
        s = np.concatenate([ix[e] for e in rng.choice(eps, len(eps), replace=True)])
        out.append(num[s].sum() / max(den[s].sum(), 1e-30))
    return pt, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# ----------------------------------------------------------------------------
#                                  АНАЛИЗ
# ----------------------------------------------------------------------------
def analyze(G, chg, epi, pos, task, floor_om, P, args) -> None:
    """Вся статистика считается ЗДЕСЬ, из сохранённой таблицы G.

    Никакой модели и кодека не требуется, поэтому любые изменения масок,
    порогов и слоёв пересчитываются мгновенно."""
    sets, idx = build_sets(P)
    tA, tAq, tAr, tAqr, t_um, qr_um, t_na = build_triples(P, idx)
    mA, mAq, m_um, m_qm, m_na = build_mono(P, idx)
    n_rows = G.shape[0]

    g1 = G[:, 1:1 + P].max(1)                     # лучший одиночный
    gbest = G.max(1)                              # полный доступный выигрыш
    g1m, gbm = g1.mean(), gbest.mean()

    # ПОРОГ. По вмешательству (все строки одной позиции делят медиану), но не
    # ниже измеренного пола ИМЕННО ДЛЯ Omega.
    tau = np.empty(n_rows)
    for p_ in np.unique(pos):
        m = pos == p_
        tau[m] = max(1e-8, args.tau_rel * float(np.median(g1[m])), floor_om)

    om = G[:, tAq] + G[:, tAr] - G[:, tA] - G[:, tAqr]
    mo = G[:, mAq] - G[:, mA]
    cm = chg[:, None]
    nz = (qr_um[None, :] & ~cm) == 0              # содержательные тройки
    inside = (t_um[None, :] & ~cm) == 0           # ещё и A внутри support
    mnz = (m_qm[None, :] & ~cm) == 0              # содержательные пары

    # СЛОИ ПО СИЛЕ. Маска top-4 ОБЯЗАНА пересекаться с nz: у неизменённой
    # позиции одиночный выигрыш ровно ноль, поэтому при support меньше четырёх
    # она попадает в top-4 и приносит тождественно нулевые тройки.
    t4 = np.argsort(-G[:, 1:1 + P], 1)[:, :4]
    top_m = np.zeros(n_rows, np.int64)
    for j in range(4):
        top_m |= (1 << t4[:, j].astype(np.int64))
    hi = nz & ((qr_um[None, :] & ~top_m[:, None]) == 0)
    hi_str = hi & inside
    lo_m = nz & ~hi

    tt = tau[:, None]
    viol, mviol = om < -tt, mo < -tt

    print("=" * 78)
    print("K-4a4. ПРЯМОЙ ТЕСТ СУБМОДУЛЯРНОСТИ (всё на КВАДРАТЕ ошибки)")
    print("=" * 78)
    print(f"вмешательств {n_rows}, эпизодов {len(np.unique(epi))}, "
          f"позиций {len(np.unique(pos))}, троек {om.shape[1]}, "
          f"пар монотонности {mo.shape[1]}")
    print(f"масштаб: лучший одиночный {g1m:.3e}, полный доступный {gbm:.3e}")
    print(f"пол для Omega (float32 против float64) {floor_om:.3e}, "
          f"порог tau в долях g1 {args.tau_rel:g}, "
          f"медианный tau {np.median(tau):.3e}\n")

    def rate(mask_num, mask_den, name, width=38):
        n = (viol & mask_num).sum(1).astype(np.float64)
        d = mask_den.sum(1).astype(np.float64)
        pt, lo, hi_ = cluster_ci(n, d, epi)
        print(f"  {name:>{width}} {pt:8.2%} [{lo:.2%}, {hi_:.2%}]"
              f"   троек/вмеш. {d.mean():8.1f}")
        return pt

    print("ДОЛЯ НАРУШЕНИЙ СУБМОДУЛЯРНОСТИ")
    r_all = rate(nz, nz, "содержательные (q,r изменены)")
    r_str = rate(inside, inside, "строго (и A внутри support)")
    r_hi = rate(hi, hi, "СОДЕРЖ. пары из top-4")
    r_hs = rate(hi_str, hi_str, "СОДЕРЖ. top-4, строго (A внутри)")
    r_lo = rate(lo_m, lo_m, "остальные содержательные")
    print(f"  {'для сравнения, по ВСЕМ тройкам':>38} "
          f"{(viol.sum() / viol.size):8.2%}   <- разбавлено нулями, "
          f"содержательных {nz.mean():.2%}")

    nhi = hi.sum(1)
    print(f"\n  содержательных top-4 троек на вмешательство: "
          f"среднее {nhi.mean():.1f}, медиана {np.median(nhi):.0f}, "
          f"квартили {np.percentile(nhi, 25):.0f}/{np.percentile(nhi, 75):.0f}")
    print(f"  вмешательств без единой содержательной top-4 тройки: "
          f"{(nhi == 0).mean():.1%}")

    print("\nЗНАК И ВЕЛИЧИНА Omega ПО СЛОЯМ (Omega > 0 = убывающая отдача)")
    print(f"  {'слой':>34}{'средн. Om':>11}{'средн.|Om|':>11}"
          f"{'асимметрия':>12}{'|Om|/tau':>10}")
    asym = {}
    for nm, key, msk in (("все содержательные", "all", nz),
                         ("пары из top-4", "hi", hi),
                         ("top-4, строго", "hs", hi_str),
                         ("остальные содерж.", "lo", lo_m)):
        n_ = np.maximum(msk.sum(1), 1)
        mo_ = ((om * msk).sum(1) / n_).mean()
        ao_ = ((np.abs(om) * msk).sum(1) / n_).mean()
        rt = ((np.abs(om) * msk).sum(1) / n_ / tau).mean()
        asym[key] = mo_ / max(ao_, 1e-30)
        print(f"  {nm:>34}{mo_ / g1m:>+10.2%}{ao_ / g1m:>11.2%}"
              f"{asym[key]:>+12.2f}{rt:>10.0f}")
    print("""  АСИММЕТРИЯ = средн.Om / средн.|Om|, лежит в [-1, +1] и от масштаба
  НЕ зависит. Это единственный законный способ сравнить слои, когда их |Om|
  различаются: доля нарушений при разном разбросе несопоставима, а знак
  распределения — сопоставим. +1 означает «всегда избыточность», -1 —
  «всегда комплементарность», 0 — симметрия.""")

    print("\nВЕЛИЧИНА НАРУШЕНИЯ (только нарушения, в долях одиночного выигрыша)")
    for nm, msk in (("содержательные", nz), ("пары из top-4", hi)):
        v = np.abs(om[viol & msk]) / g1m
        if v.size:
            print(f"  {nm:>38}: медиана {np.median(v):.2%}, "
                  f"90-й {np.percentile(v, 90):.2%}, "
                  f"99-й {np.percentile(v, 99):.2%}, макс {v.max():.2%}")

    print("\nСВИП ПОРОГА: доля нарушений при разных tau/g1")
    print(f"  {'tau/g1':>10}{'содержательные':>18}{'top-4':>12}"
          f"{'монотонность':>16}")
    for tr in TAU_SWEEP:
        t2 = np.empty(n_rows)
        for p_ in np.unique(pos):
            m = pos == p_
            t2[m] = max(floor_om, tr * float(np.median(g1[m])))
        v2, mv2 = om < -t2[:, None], mo < -t2[:, None]
        print(f"  {tr:>10.0e}{(v2 & nz).sum() / max(nz.sum(), 1):>17.2%}"
              f"{(v2 & hi).sum() / max(hi.sum(), 1):>12.2%}"
              f"{(mv2 & mnz).sum() / max(mnz.sum(), 1):>16.2%}")

    print("\nМОНОТОННОСТЬ")
    n = (mviol & mnz).sum(1).astype(np.float64)
    d = mnz.sum(1).astype(np.float64)
    pt, lo, hi_ = cluster_ci(n, d, epi)
    print(f"  доля нарушений (q изменена): {pt:.2%} [{lo:.2%}, {hi_:.2%}]")
    ins_m = (m_um[None, :] & ~cm) == 0
    n2 = (mviol & ins_m).sum(1).astype(np.float64)
    pt2, lo2, hi2 = cluster_ci(n2, ins_m.sum(1).astype(np.float64), epi)
    print(f"  строго (A внутри support):   {pt2:.2%} [{lo2:.2%}, {hi2:.2%}]")
    dmg = np.abs(mo[mviol & mnz])
    if dmg.size:
        print(f"  ВЕЛИЧИНА УЩЕРБА, в долях одиночного выигрыша: "
              f"медиана {np.median(dmg) / g1m:.2%}, "
              f"90-й {np.percentile(dmg, 90) / g1m:.2%}, "
              f"99-й {np.percentile(dmg, 99) / g1m:.2%}")
        print(f"  то же в долях полного доступного разрыва: "
              f"медиана {np.median(dmg) / gbm:.2%}, "
              f"99-й {np.percentile(dmg, 99) / gbm:.2%}")

    print("\nДОЛЯ НАРУШЕНИЙ ПО РАЗМЕРУ A (только содержательные тройки)")
    print(f"  {'|A|':>5}{'содержательные':>18}{'строго':>12}{'троек':>12}")
    for ka in range(int(t_na.max()) + 1):
        sl = t_na == ka
        a1 = (viol[:, sl] & nz[:, sl]).sum() / max(nz[:, sl].sum(), 1)
        a2 = (viol[:, sl] & inside[:, sl]).sum() / max(inside[:, sl].sum(), 1)
        print(f"  {ka:>5}{a1:>17.2%}{a2:>12.2%}{int(nz[:, sl].sum()):>12}")

    print("\nМАКРО-СТАТИСТИКА ПО ВМЕШАТЕЛЬСТВАМ (по содержательным тройкам)")
    mac = (viol & nz).sum(1) / np.maximum(nz.sum(1), 1)
    ok = nz.sum(1) > 0
    print(f"  среднее {mac[ok].mean():.2%}, медиана {np.median(mac[ok]):.2%}, "
          f"90-й проц. {np.percentile(mac[ok], 90):.2%}, "
          f"макс {mac[ok].max():.2%}")
    print(f"  доля вмешательств хотя бы с одним содержательным нарушением: "
          f"{((viol & nz).sum(1) > 0).mean():.1%}")
    mas = (viol & inside).sum(1) / np.maximum(inside.sum(1), 1)
    print(f"  строго: среднее {mas.mean():.2%}, медиана {np.median(mas):.2%}, "
          f"90-й проц. {np.percentile(mas, 90):.2%}")

    # ЛОКАЛЬНОЕ ПАРНОЕ ОТНОШЕНИЕ. ОПИСАТЕЛЬНАЯ статистика, НЕ submodularity
    # ratio и НЕ основание для гарантии жадного отбора: тот определяется как
    # минимум по допустимым парам множеств и требует монотонности функции,
    # которая здесь нарушается. Никакого 1 - exp(-gamma) не вычисляется.
    den = G[:, tAqr] - G[:, tA]
    okd = (den > tt) & nz
    ratio = 1.0 + om[okd] / den[okd]
    print("\nЛОКАЛЬНОЕ ПАРНОЕ ОТНОШЕНИЕ [m(q|A)+m(r|A)] / m(qr|A) — ОПИСАТЕЛЬНО")
    print(f"  (не submodularity ratio Das-Kempe; теоретической гарантии из него "
          f"не следует)\n  считается по {ratio.size} тройкам со знаменателем "
          f"выше tau")
    for q in (1, 5, 25, 50, 75, 95):
        print(f"    {q:>2}-й процентиль {np.percentile(ratio, q):>8.3f}")
    print(f"    доля ниже 1: {(ratio < 1).mean():.1%}")

    if task is not None and len(np.unique(task)) > 1:
        print("\nПО ЗАДАЧАМ (12 самых частых)")
        u, c = np.unique(task, return_counts=True)
        print(f"  {'задача':>52}{'вмеш.':>7}{'эпиз':>6}{'содерж.':>10}"
              f"{'top-4':>9}")
        for t in u[np.argsort(-c)][:12]:
            m = task == t
            if m.sum() < 20:
                continue
            a1 = (viol[m] & nz[m]).sum() / max(nz[m].sum(), 1)
            a2 = (viol[m] & hi[m]).sum() / max(hi[m].sum(), 1)
            print(f"  {str(t)[:52]:>52}{int(m.sum()):>7}"
                  f"{len(np.unique(epi[m])):>6}{a1:>10.2%}{a2:>9.2%}")

    print("\n" + "=" * 78)
    print("ВЫВОД")
    print("=" * 78)
    print(f"  содержательные тройки: {r_all:.2%} нарушений")
    print(f"  пары из top-4:         {r_hi:.2%} нарушений, "
          f"среднее Omega {((om * hi).sum(1) / np.maximum(hi.sum(1), 1)).mean() / g1m:+.2%}")
    if r_all > 0.30:
        print("""
  ГЛОБАЛЬНО функция НЕ субмодулярна: знак взаимодействия произвольной
  содержательной пары близок к случайному. Формулировка «приближённая
  субмодулярность» недоступна.""")
    print(f"  асимметрия: top-4 {asym['hi']:+.2f}, остальные {asym['lo']:+.2f}"
          f" (величина безразмерная, от масштаба слоя не зависит)")
    if asym["hi"] > 0.3 and asym["lo"] < -0.3:
        print("""
  ЗНАК ВЗАИМОДЕЙСТВИЯ ПРЕДСКАЗЫВАЕТСЯ СИЛОЙ ПОЗИЦИЙ. Сильные пары
  систематически ИЗБЫТОЧНЫ, слабые — систематически КОМПЛЕМЕНТАРНЫ. Это не
  «убывающая отдача с шумом», а две разные задачи в одном наборе, и
  set-aware router нужен для обеих: не брать дублирующие сильные позиции и
  не упускать дополняющие слабые.""")
    elif asym["hi"] > 0.3:
        print("""
  Сильные пары систематически избыточны, у остальных знак близок к
  симметричному. Допустима формулировка «сильные позиции систематически
  перекрываются, тогда как взаимодействие произвольных пар знакопеременно».""")
    else:
        print("""
  Слой сильных позиций не отличается по знаку: взаимодействия знакопеременны
  и там. Тогда set-aware router нужен не только чтобы устранять избыточность,
  но и чтобы находить редкие комплементарные пары.""")
    print("""
Направление это НЕ закрывает: оракульная разреженность и провал независимого
ранжирования от субмодулярности не зависят. Закрывается лишь возможность
опереть новизну на субмодулярность.
МОНОТОННОСТЬ. Нарушения означают, что добавление позиции способно ухудшить
результат: оракулу нужно право отказа, router нельзя учить «чем больше, тем
лучше», а L_BCE по changed-support не может быть основной целью.""")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-npz", default=None,
                    help="пересчитать статистику из сохранённых таблиц, "
                         "без модели и кодека")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n-obs", type=int, default=96)
    ap.add_argument("--n-ep", type=int, default=48)
    ap.add_argument("--max-pos", type=int, default=0)
    ap.add_argument("--set-block", type=int, default=32)
    ap.add_argument("--tau-rel", type=float, default=1e-3)
    ap.add_argument("--rank-lo", type=int, default=1)
    ap.add_argument("--rank-hi", type=int, default=5)
    ap.add_argument("--pos-offset", type=int, default=3)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--embodiment", type=int, default=0)
    ap.add_argument("--center-crop", action="store_true", default=True)
    ap.add_argument("--tiled", action="store_true", default=True)
    ap.add_argument("--source", default="lerobot")
    ap.add_argument("--flip", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump", default="logs/k4a4_submodularity.npz")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    selftest()

    if args.from_npz:
        d = np.load(args.from_npz, allow_pickle=True)
        print(f"пересчёт из {args.from_npz}: commit {d['commit']}, "
              f"seed {d['seed']}\n")
        analyze(d["G"], d["chg"], d["epi"], d["pos"],
                d["task"] if "task" in d else None,
                float(d["floor_om"]), int(d["P"]), args)
        return

    if not args.ckpt:
        raise SystemExit("нужен --ckpt или --from-npz")

    import torch

    from k1_residual_cost import latent_from_codes, projected_codebooks
    from k3_bar_suffix_repair import (MAX_ACTION_Q, STATE_Q01, STATE_Q99,
                                      build_batch, load_lerobot)

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         text=True).strip()
    except Exception:
        commit = "unknown"
    print(f"commit {commit}, seed {args.seed}\n")

    sys.path.insert(0, os.path.abspath(args.root))
    import copy
    import importlib.util

    import actioncodec  # noqa: F401

    sp = importlib.util.spec_from_file_location(
        "ac_vla_tok", os.path.join(os.path.abspath(args.root), "utils",
                                   "vla_tokenizer.py"))
    mm = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mm)
    proc = mm.VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")
    tok = proc.action_processor.to(args.device).eval()
    L, P = tok.num_quantizers, tok.n_tokens_per_quantizer
    cfg = tok.config.embodiment_config[
        list(tok.config.embodiment_config.keys())[args.embodiment]]
    T, D_act = int(cfg["freq"] * cfg["duration"]), cfg["action_dim"]
    print(f"dtype кодека при загрузке: {next(tok.parameters()).dtype}")
    tok32 = copy.deepcopy(tok).float().eval()
    E = projected_codebooks(tok32, args.device)

    IM1, IM2, ST_RAW, A_, PREV, tasks, EPI = load_lerobot(
        args.n_obs, T, n_ep=args.n_ep, seed=args.seed)
    A_ = np.asarray(A_, np.float32).copy()
    A_[..., :-1] = A_[..., :-1] / MAX_ACTION_Q[:-1]
    A_[..., -1] = -A_[..., -1]
    a_true = torch.from_numpy(np.clip(A_, -1.0, 1.0)).to(args.device)
    scale = float(a_true.max() - a_true.min())
    B = len(A_)
    st = ((ST_RAW - STATE_Q01) / (STATE_Q99 - STATE_Q01) * 2.0 - 1.0).astype(np.float32)

    from smolvla.bar import SmolVLABlockwiseAR

    model = SmolVLABlockwiseAR.from_pretrained(
        args.ckpt, trust_remote_code=True, dtype=torch.bfloat16,
        token_budget=P * L, num_blocks=L,
        action_vocab_size=tok.vocab_size).to(args.device).eval()
    bs, nb = model.block_size, model.num_blocks

    batch = build_batch(IM1, IM2, tasks, st, proc, args, args.device)
    with torch.no_grad():
        _, vlen, VLM, _ = model._build_vlm_inputs_embeds(
            input_ids=batch["input_ids"], inputs_embeds=None,
            pixel_values=batch.get("pixel_values"),
            pixel_attention_mask=batch.get("pixel_attention_mask"),
            image_hidden_states=None)

    def blk(hist):
        alen = bs + (0 if hist is None else hist.shape[1])
        apos = model._build_action_pos_ids_strided(
            batch_size=B, base_pos=vlen, action_seq_len=alen,
            device=VLM.device, position_offset=args.pos_offset)
        pids = model._build_joint_position_ids(
            batch_size=B, vlm_seq_len=vlen, action_pos_ids=apos, device=VLM.device)
        return model._predict_next_block_logits(
            vlm_inputs_embeds=VLM, attention_mask=batch.get("attention_mask"),
            history_tokens=hist, position_ids=pids).float()

    def gen(hist, n):
        for _ in range(n):
            c = blk(hist).argmax(-1)
            hist = c if hist is None else torch.cat([hist, c], 1)
        return hist

    def dec_with(m, h):
        out = []
        for i in range(0, len(h), args.chunk):
            out.append(m._decode(h[i:i + args.chunk], args.embodiment,
                                 None)[0][..., :D_act])
        return torch.cat(out)

    def sq_with(m, h, ref):
        d = (dec_with(m, h)[:, :args.window]
             - ref[:, :args.window]).abs()[..., :D_act - 1]
        return d.flatten(1).pow(2).mean(-1) / scale ** 2

    sets, idx = build_sets(P)
    print(f"наборов размера <=4 (с пустым): {len(sets)}")

    def gtable(m, EE, dt, stale, z_ref, aref=None):
        """Таблица G(S) для всех наборов, в заданной точности."""
        hs = latent_from_codes(EE, stale).to(dt)
        hr = latent_from_codes(EE, z_ref).to(dt)
        ref = dec_with(m, hr) if aref is None else aref
        e0 = sq_with(m, hs, ref)
        Gt = torch.zeros(hs.shape[0], len(sets), dtype=torch.float64,
                         device=hs.device)
        for i in range(1, len(sets), args.set_block):
            blockS = sets[i:i + args.set_block]
            hh = hs.unsqueeze(0).repeat(len(blockS), 1, 1, 1)
            for j, S in enumerate(blockS):
                hh[j][:, list(S)] = hr[:, list(S)]
            ee = sq_with(m, hh.reshape(-1, P, hs.shape[-1]),
                         ref.repeat(len(blockS), 1, 1))
            Gt[:, i:i + len(blockS)] = (e0.repeat(len(blockS)) - ee).reshape(
                len(blockS), -1).T.double()
        return Gt

    rng = torch.Generator(device=args.device).manual_seed(1)
    ar = torch.arange(B, device=args.device)
    n_pos = args.max_pos or P

    def make_stale(p_):
        u = lg0[:, p_].topk(args.rank_hi, -1).indices[
            ar, torch.randint(args.rank_lo, args.rank_hi, (B,),
                              generator=rng, device=args.device)]
        c0 = z_ref[:, :, 0].clone()
        c0[:, p_] = u
        c1 = blk(c0).argmax(-1)
        z_old = torch.stack([c0, c1,
                             blk(torch.cat([c0, c1], 1)).argmax(-1)], -1)
        s = z_old.clone()
        s[:, :, 0] = z_ref[:, :, 0]
        return s

    with torch.no_grad():
        z_ref = gen(None, nb).reshape(-1, L, P).transpose(1, 2)
        a_ref = dec_with(tok32, latent_from_codes(E, z_ref))
        lg0 = blk(None)

        # ---------- ПОЛ ДЛЯ Omega, а не для одиночных выигрышей ----------
        # Omega — разность ЧЕТЫРЁХ величин, её численный пол заведомо выше.
        # Меряем сравнением полных таблиц G во float32 и float64.
        tok64 = copy.deepcopy(tok).double().eval()
        E64 = projected_codebooks(tok64, args.device)
        st0 = make_stale(0)
        G32 = gtable(tok32, E, torch.float32, st0, z_ref).cpu().numpy()
        G64 = gtable(tok64, E64, torch.float64, st0, z_ref).cpu().numpy()
        tA, tAq, tAr, tAqr, _, _, _ = build_triples(P, idx)
        d_om = np.abs((G32[:, tAq] + G32[:, tAr] - G32[:, tA] - G32[:, tAqr])
                      - (G64[:, tAq] + G64[:, tAr] - G64[:, tA] - G64[:, tAqr]))
        g1_0 = G64[:, 1:1 + P].max(1)
        floor_om = 3.0 * float(d_om.max())
        print(f"\nЧИСЛЕННЫЙ ПОЛ ДЛЯ Omega (float32 против float64), "
              f"вмешательство 0")
        print(f"  максимум расхождения {d_om.max():.3e} "
              f"({d_om.max() / g1_0.mean():.4%} одиночного выигрыша)")
        print(f"  99-й процентиль      {np.percentile(d_om, 99):.3e}")
        print(f"  для сравнения, пол по одиночным выигрышам "
              f"{np.abs(G32[:, 1:1 + P] - G64[:, 1:1 + P]).max():.3e}")
        print(f"  пол 3*max = {floor_om:.3e}, порог {args.tau_rel:g}*g1 "
              f"= {args.tau_rel * np.median(g1_0):.3e}\n")
        del tok64, E64, G64
        torch.cuda.empty_cache()

        Gs, chgs, poss = [], [], []
        for p_ in range(n_pos):
            stale = make_stale(p_) if p_ else st0
            Gt = (G32 if p_ == 0 else
                  gtable(tok32, E, torch.float32, stale, z_ref,
                         a_ref).cpu().numpy())
            Gs.append(Gt.astype(np.float32))
            diff = (stale != z_ref).any(-1)
            chgs.append((diff.int()
                         * (1 << torch.arange(P, device=args.device))
                         ).sum(-1).cpu().numpy())
            poss.append(np.full(B, p_))
            print(f"  позиция {p_ + 1}/{n_pos} готова", flush=True)

    G = np.concatenate(Gs)
    chg = np.concatenate(chgs)
    pos = np.concatenate(poss)
    epi = np.tile(EPI, n_pos)
    task = np.tile(np.asarray(tasks), n_pos)

    if args.dump:
        os.makedirs(os.path.dirname(args.dump) or ".", exist_ok=True)
        np.savez_compressed(args.dump, G=G, chg=chg, epi=epi, pos=pos,
                            task=task, floor_om=floor_om, P=P,
                            commit=commit, seed=args.seed,
                            tau_rel=args.tau_rel, window=args.window)
        print(f"\nтаблицы сохранены целиком: {args.dump} "
              f"({G.nbytes / 1e6:.1f} МБ до сжатия)\n")

    analyze(G, chg, epi, pos, task, floor_om, P, args)


if __name__ == "__main__":
    main()
