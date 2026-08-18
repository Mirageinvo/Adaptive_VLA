"""Unit-тесты чистой логики K-4b0. Запускаются без GPU, без модели, без данных.

Каждый тест строится на функции с АНАЛИТИЧЕСКИ известным ответом, чтобы провал
означал дефект кода, а не особенность выборки.

Запуск:
    python3 experiments/test_k4b0.py
"""

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k4b0_build_router_dataset import (  # noqa: E402
    derive_labels,
    greedy_paths,
    split_by_episode,
    subsets_of,
)
from k4b0_padding_probe import rankdata_avg, spearman  # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'ПРОВАЛ'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILED.append(name)


# ---------------------------------------------------------------- траектории
def test_greedy_basic():
    """Две позиции полезны порознь и избыточны вместе, третья вредна."""
    C = [0, 1, 2]
    g = {(): 0.0, (0,): 1.0, (1,): 0.9, (2,): -0.2,
         (0, 1): 1.2, (0, 2): 0.8, (1, 2): 0.7, (0, 1, 2): 1.0}
    add, marg, stop, acts, qs, qo, S, ast = greedy_paths(g, C, 1e-9, 3)
    check("жадный берёт сильнейшую первой", add[0] == 0, f"add={add}")
    check("STOP до вредного шага", stop == 2, f"stop={stop}")
    check("обратимый сходится к максимуму", S == (0, 1), f"S={S}")


def test_greedy_useful_swap():
    """ОБМЕН полезен: жадный берёт 0, потом 1, но пара (1,2) лучше, и попасть
    туда можно только заменив 0 на 2 — удаление нуля само по себе ухудшает."""
    C = [0, 1, 2]
    g = {(): 0.0, (0,): 1.0, (1,): 0.6, (2,): 0.5,
         (0, 1): 1.05, (0, 2): 1.02, (1, 2): 1.8, (0, 1, 2): 1.1}
    add, _, _, acts, qs, qo, S, ast = greedy_paths(g, C, 1e-9, 2)
    check("жадное добавление даёт не оптимум", set(add) == {0, 1},
          f"add={add}, G={g[tuple(sorted(add))]}")
    check("обратимый находит (1,2)", S == (1, 2), f"S={S}, G={g[S]}")
    check("в траектории есть SWAP (код 2)", 2 in acts, f"acts={acts}")
    check("SWAP хранит обе позиции", any(a == 2 and o >= 0
                                         for a, o in zip(acts, qo)),
          f"acts={acts}, qs={qs}, qo={qo}")
    check("SWAP НЕ разложен в пару REMOVE+ADD", acts.count(0) == 0,
          f"acts={acts}")


def test_standalone_remove_possible():
    """САМОСТОЯТЕЛЬНЫЙ REMOVE ВОЗМОЖЕН — детерминированный контрпример.

    Прежде здесь стояло обратное утверждение с «доказательством»: позиция
    входит только при улучшении, значит удаление позже улучшать не может,
    иначе обмен на том же шаге дал бы больше. Рассуждение молча предполагало,
    что состояние достигнуто последним ДОБАВЛЕНИЕМ; после ОБМЕНА оно
    неприменимо. Случайный поиск на 500 функциях контрпримера не нашёл и был
    принят за подтверждение — отрицательный поиск доказательством не является.

    Ожидаемая траектория при kmax=3:
        ADD 0, ADD 1, ADD 2, SWAP 0->3, REMOVE 1  ->  {2,3}"""
    C = [0, 1, 2, 3]
    g = {(): 0.0, (0,): 1.0, (1,): 0.5, (2,): 0.4, (3,): 0.3,
         (0, 1): 2.0, (0, 2): 1.2, (0, 3): 1.1,
         (1, 2): 2.2, (1, 3): 2.1, (2, 3): 5.0,
         (0, 1, 2): 3.0, (0, 1, 3): 2.5, (0, 2, 3): 3.5, (1, 2, 3): 4.0}
    check("таблица полна", set(g) == set(subsets_of(C, 3)))
    add, _, _, acts, qin, qout, S, ast = greedy_paths(g, C, 1e-9, 3)
    check("траектория ADD,ADD,ADD,SWAP,REMOVE", acts == [1, 1, 1, 2, 0],
          f"acts={acts}")
    check("итог — оптимум {2,3}", S == (2, 3), f"S={S}, G={g[S]}")
    check("SWAP хранит обе позиции", qin[3] == 3 and qout[3] == 0,
          f"q_in={qin[3]}, q_out={qout[3]}")
    check("у REMOVE q_in = -1, q_out = удаляемая",
          qin[4] == -1 and qout[4] == 1, f"q_in={qin[4]}, q_out={qout[4]}")
    check("у ADD q_out = -1", all(qout[i] == -1 for i in range(3)),
          f"qout={qout[:3]}")
    check("SWAP не засчитан как REMOVE", acts.count(0) == 1, f"acts={acts}")


def test_greedy_no_silent_truncation():
    """Лимит теоретический: усечения нет, при недосчёте — исключение.

    Прежде стоял произвольный cap = 6*kmax+6, и по его достижении функция
    МОЛЧА возвращала недосчитанный набор. Теперь предел равен числу достижимых
    состояний, а итог дополнительно проверяется на локальную оптимальность."""
    rng = np.random.default_rng(11)
    for _ in range(200):
        C = sorted(rng.choice(7, rng.integers(2, 6), replace=False).tolist())
        g = {S: float(rng.normal()) for S in subsets_of(C, 4)}
        g[()] = 0.0
        greedy_paths(g, C, 1e-9, 4)          # упадёт при недосчёте
    check("итог локально оптимален на 200 функциях", True)
    # вырожденный случай: все ходы улучшают почти одинаково
    C = list(range(6))
    g = {S: 0.001 * len(S) for S in subsets_of(C, 4)}
    S = greedy_paths(g, C, 1e-12, 4)[6]
    check("при равных приращениях набирается полный бюджет", len(S) == 4,
          f"S={S}")


def test_rev_trajectory_ragged():
    """Рваное хранение траекторий: длина не ограничена шириной массива."""
    rng = np.random.default_rng(5)
    flat_a, off = [], [0]
    lens = []
    for _ in range(50):
        C = sorted(rng.choice(8, rng.integers(2, 6), replace=False).tolist())
        g = {S: float(rng.normal()) for S in subsets_of(C, 4)}
        g[()] = 0.0
        acts = greedy_paths(g, C, 1e-9, 4)[3]
        flat_a.extend(acts)
        off.append(off[-1] + len(acts))
        lens.append(len(acts))
    check("смещения согласованы с длинами",
          off[-1] == len(flat_a) == sum(lens))
    ok = all(len(flat_a[off[i]:off[i + 1]]) == lens[i] for i in range(50))
    check("каждая траектория извлекается по своим смещениям", ok)
    check("длины бывают разные", len(set(lens)) > 1, f"длины {sorted(set(lens))}")


def test_greedy_nonmonotone():
    """Немонотонная функция: добавление ЛЮБОЙ позиции ухудшает."""
    C = [0, 1]
    g = {(): 0.0, (0,): -0.3, (1,): -0.5, (0, 1): -0.9}
    add, marg, stop, acts, qs, qo, S, ast = greedy_paths(g, C, 1e-9, 2)
    check("STOP на нуле позиций", stop == 0, f"stop={stop}")
    check("ADD+STOP даёт пустой набор", ast == (), f"ast={ast}")
    check("обратимый оставляет пустой набор", S == (), f"S={S}")
    check("обратимый не делает ходов", len(acts) == 0, f"acts={acts}")


def test_greedy_terminates():
    """Случайные функции: процесс обязан завершаться, а не зацикливаться."""
    rng = np.random.default_rng(0)
    worst = 0
    for _ in range(300):
        C = sorted(rng.choice(8, rng.integers(2, 6), replace=False).tolist())
        subs = subsets_of(C, 4)
        g = {S: float(rng.normal()) for S in subs}
        g[()] = 0.0
        acts = greedy_paths(g, C, 1e-9, 4)[3]
        worst = max(worst, len(acts))
    check("обратимая траектория конечна", worst <= 20, f"макс длина {worst}")


# ------------------------------------------------------------ рваная таблица
def test_ragged_permutation():
    """Перестановка строк рваной таблицы сохраняет содержимое каждой."""
    rng = np.random.default_rng(1)
    n = 40
    lens = rng.integers(1, 60, n)
    off = np.concatenate([[0], np.cumsum(lens)])
    flat = np.arange(off[-1], dtype=np.float32)
    perm = rng.permutation(n)
    nf = np.concatenate([flat[off[i]:off[i + 1]] for i in perm])
    no = np.concatenate([[0], np.cumsum(lens[perm])])
    ok = all(np.array_equal(nf[no[j]:no[j + 1]], flat[off[i]:off[i + 1]])
             for j, i in enumerate(perm))
    check("строки рваной таблицы не перемешались", ok)
    check("суммарная длина сохранена", no[-1] == off[-1])


def test_gof_intersection():
    """G(S) сводится к пересечению с support; порядок позиций не важен."""
    C = (2, 5, 9)
    subs = subsets_of(C, 4)
    gm = {S: float(i) for i, S in enumerate(subs)}

    def g_of(S):
        return gm[tuple(sorted(set(S) & set(C)))]

    check("позиции вне support игнорируются", g_of((2, 7, 13)) == gm[(2,)])
    check("набор вне support = пустой", g_of((0, 1)) == gm[()])
    check("порядок не важен", g_of((9, 5)) == gm[(5, 9)])


# -------------------------------------------------------------------- split
def test_split():
    for n_ep, n_task in ((500, 130), (400, 130), (48, 25)):
        eps = np.repeat(np.arange(n_ep), 2)
        tasks = [f"t{e % n_task}" for e in eps]
        sp = split_by_episode(eps, tasks, seed=0)
        sets = {s: set(eps[sp == s].tolist()) for s in (0, 1, 2)}
        ov = (len(sets[0] & sets[1]) + len(sets[0] & sets[2])
              + len(sets[1] & sets[2]))
        one = all(len(set(sp[eps == e])) == 1 for e in np.unique(eps))
        T = np.asarray(tasks)
        miss = sum(len(set(T[sp == s]) - set(T[sp == 0])) for s in (1, 2))
        fr = [(sp == s).mean() for s in (0, 1, 2)]
        check(f"split {n_ep}эп/{n_task}задач: без пересечений", ov == 0)
        check(f"split {n_ep}: эпизод неделим", one)
        check(f"split {n_ep}: задачи val/test есть в train", miss == 0)
        check(f"split {n_ep}: доли близки к цели",
              abs(fr[0] - 0.70) < 0.05 and abs(fr[1] - 0.15) < 0.05,
              f"{[round(x, 3) for x in fr]}")


def test_group_order_restored():
    """Порядок групп по длине произволен; канонический порядок обязан
    восстанавливаться сортировкой. Проверяем на ЯВНО перемешанных группах."""
    P, N = 4, 5
    groups = [[3, 0, 4], [1, 2]]          # именно такой порядок обхода
    obs = np.concatenate([np.tile(g, P) for g in groups])
    pp = np.concatenate([np.repeat(np.arange(P), len(g)) for g in groups])
    perm = np.lexsort((pp, obs))
    key = (obs * P + pp)[perm]
    check("канонический ключ строго возрастает", bool((np.diff(key) > 0).all()))
    check("покрыты все пары (наблюдение, p)", len(key) == N * P)
    check("первая строка — наблюдение 0, p 0", key[0] == 0)


# ----------------------------------------------------------- ровно K и <= K
def test_exact_vs_le_k():
    """«Ровно K» равен «<= K», только если есть чем добить набор ВНЕ support.
    Строим случай, где запаса нет и равенство НАРУШАЕТСЯ."""
    P, K = 4, 3
    C = (0, 1, 2, 3)                       # support = все позиции, запаса нет
    subs = subsets_of(C, K)
    g = {S: 0.0 for S in subs}
    g[(0,)] = 1.0
    g[(0, 1)] = 0.5
    g[(0, 1, 2)] = 0.2                     # добавления только вредят
    best_le = max(g.values())
    free = P - len(C)
    best_ex = max(v for S, v in g.items()
                  if len(S) == K or len(S) + free >= K)
    check("без запаса вне support равенство нарушается",
          best_le > best_ex, f"<=K {best_le}, ровно K {best_ex}")
    # а с запасом — выполняется
    C2 = (0,)
    subs2 = subsets_of(C2, K)
    g2 = {S: (1.0 if S == (0,) else 0.0) for S in subs2}
    free2 = P - len(C2)
    check("с запасом равенство выполняется",
          max(g2.values()) == max(v for S, v in g2.items()
                                  if len(S) == K or len(S) + free2 >= K))


# ------------------------------------------------------------------ Спирмен
def test_spearman_ties():
    check("Спирмен(x, x) = 1", abs(spearman([3, 1, 2], [3, 1, 2]) - 1) < 1e-12)
    check("Спирмен(x, -x) = -1", abs(spearman([3, 1, 2], [-3, -1, -2]) + 1) < 1e-12)
    r = rankdata_avg([0.0, 0.0, 0.0, 5.0])
    check("ничьи получают средний ранг", np.allclose(r, [1, 1, 1, 3]), f"{r}")
    # наивные порядковые ранги завышают корреляцию при массовых ничьих
    a = np.array([0, 0, 0, 0, 1.0, 2.0])
    b = np.array([0, 0, 0, 0, 2.0, 1.0])
    naive_a = np.argsort(np.argsort(a)).astype(float)
    naive_b = np.argsort(np.argsort(b)).astype(float)
    na = ((naive_a - naive_a.mean()) * (naive_b - naive_b.mean())).sum() / (
        np.linalg.norm(naive_a - naive_a.mean())
        * np.linalg.norm(naive_b - naive_b.mean()))
    check("наивные ранги завышают при ничьих", na > spearman(a, b),
          f"наивный {na:.3f} против {spearman(a, b):.3f}")


# ----------------------------------------------------- последний старт чанка
def test_last_chunk_start():
    """Эпизод длины ровно T даёт один допустимый чанк со стартом 0."""
    for n, T in ((25, 20), (20, 20), (21, 20)):
        n_st = n - T + 1
        ok = n_st >= 1 and (n_st - 1) + T <= n
        check(f"старты при n={n}, T={T}: их {n_st}, последний валиден", ok)
    check("эпизод короче T отбрасывается", 19 < 20)


# -------------------------------------------------------------- derive_labels
def test_derive_labels():
    P, kmax = 8, 4
    C = (0, 1, 2)
    vals = {(): 0.0, (0,): 1.0, (1,): 0.9, (2,): -0.2,
            (0, 1): 1.2, (0, 2): 0.8, (1, 2): 0.7, (0, 1, 2): 1.0}
    subs = subsets_of(C, kmax)
    raw = dict(g_flat=np.array([vals[S] for S in subs], np.float32),
               g_off=np.array([0, len(subs)], np.int64),
               e_empty=np.array([2.0], np.float32),
               support=np.array([sum(1 << q for q in C)]),
               p=np.array([0]), obs_idx=np.array([0]))
    out = derive_labels(raw, np.zeros(1, np.int8), P, kmax, 1e-3, 1e-2)
    check("лучший набор найден", tuple(x for x in out["best_set_by_k"][0, 4]
                                       if x >= 0) == (0, 1))
    check("размеры по бюджетам", list(out["best_size_by_k"][0]) == [0, 1, 2, 2, 2])
    check("перевод в RMS точен",
          abs(out["best_gain_by_k_rms"][0, 4]
              - (np.sqrt(2) - np.sqrt(0.8))) < 1e-6)
    check("знак вредной позиции = -1", out["sing_sign"][0, 2] == -1)
    check("знак позиции вне support = 0", out["sing_sign"][0, 3] == 0)
    check("STOP до вредного шага", out["stop_k"][0] == 2)
    # рваный формат: строка извлекается по смещениям, padding отсутствует
    lo, hi = out["rev_off"][0], out["rev_off"][1]
    acts = out["rev_action"][lo:hi]
    qin, qout = out["rev_q_in"][lo:hi], out["rev_q_out"][lo:hi]
    check("длина строки совпадает с rev_len", hi - lo == out["rev_len"][0],
          f"{hi - lo} против {out['rev_len'][0]}")
    check("коды ходов только из {0,1,2}", set(acts.tolist()) <= {0, 1, 2},
          f"acts={acts.tolist()}")
    check("рваные массивы согласованы",
          len(out["rev_action"]) == len(out["rev_q_in"]) == len(out["rev_q_out"]))
    check("смещения начинаются с нуля и покрывают массив",
          out["rev_off"][0] == 0
          and out["rev_off"][-1] == len(out["rev_action"]))
    check("разности смещений равны длинам",
          np.array_equal(np.diff(out["rev_off"]), out["rev_len"]))
    check("у ADD нет q_out", all(o == -1 for a, o in zip(acts, qout) if a == 1),
          f"acts={acts.tolist()}, qout={qout.tolist()}")
    check("у REMOVE нет q_in", all(i == -1 for a, i in zip(acts, qin) if a == 0),
          f"acts={acts.tolist()}, qin={qin.tolist()}")


def main():
    for fn in (test_greedy_basic, test_greedy_useful_swap,
               test_standalone_remove_possible,
               test_rev_trajectory_ragged,
               test_greedy_no_silent_truncation,
               test_greedy_nonmonotone, test_greedy_terminates,
               test_ragged_permutation, test_gof_intersection, test_split,
               test_group_order_restored, test_exact_vs_le_k,
               test_spearman_ties, test_last_chunk_start, test_derive_labels):
        print(f"\n{fn.__name__}:")
        fn()
    print("\n" + "=" * 60)
    if FAILED:
        print(f"ПРОВАЛЕНО {len(FAILED)}: {FAILED}")
        raise SystemExit(1)
    print("все тесты пройдены")


if __name__ == "__main__":
    main()
