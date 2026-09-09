"""K-6h, агрегатор: парное сравнение levels=1 против levels=3.

ПОЧЕМУ ОТДЕЛЬНЫЙ СКРИПТ. Каждая ячейка — свой процесс со своим файлом, потому
что среды нельзя переиспользовать. Средний успех одной ячейки из десяти
эпизодов не значит ничего; решение принимается только здесь.

ЧТО СЧИТАЕТСЯ. Эпизоды сопоставляются по ключу (suite, task_id, ensemble,
horizon, init_state_id) — то есть один и тот же эпизод при двух комплектациях
уровней. Дальше:
  * сверка парности по хешу начального наблюдения; расхождение — отказ, а не
    предупреждение, потому что тогда сравниваются разные эпизоды;
  * micro (по всем эпизодам) и macro (среднее по задачам) разности;
  * дискордантные пары и точный тест Макнемара;
  * кластерный бутстрап ПО ЗАДАЧАМ: эпизоды внутри задачи скоррелированы, и
    бутстрап по эпизодам дал бы интервал уже истинного;
  * ОДНОСТОРОННЯЯ нижняя граница — именно она отвечает на вопрос «не хуже ли
    чем на δ», тогда как пересечение двустороннего интервала с нулём означает
    лишь отсутствие доказательства разницы.

ПРАВИЛО ЧТЕНИЯ записано в докстроке соответствующего гейта ДО запуска, и оно
ТРЁХСТОРОННЕЕ: нижняя односторонняя граница выше -margin доказывает
не-худшесть; ВЕРХНЯЯ ниже -margin доказывает ухудшение более чем на margin;
между ними не доказано ничего. Прежняя версия печатала «ХУЖЕ или неясно» одной
строкой и тем склеивала второй исход с третьим.

ПОЛЕ РАЗДЕЛЕНИЯ РУК ОБОБЩЕНО. Изначально руки различались по `levels` (1 против
3). K-9d сравнивает Joint-12 с грубым выходом полной глубины и кладёт в файлы
поле `arm`. Статистика для обоих случаев одна и та же, поэтому обобщён только
ключ: --field/--test/--ref. Умолчания воспроизводят прежнее поведение
дословно, а `run_tag` вошёл в ключ ячейки, чтобы файлы K-6h и K-9d,
совпадающие по (suite, task, ens, H, init_id), не склеились в одну пару.

Запуск:
    python3 experiments/k6h_summarize.py --selftest
    python3 experiments/k6h_summarize.py --glob 'data/k6h/*.json' --margin 10
    python3 experiments/k6h_summarize.py --glob 'data/k9d/*.json' \\
        --field arm --test fast12 --ref coarse24 --margin 5
"""

import argparse
import glob as globmod
import json
import math
import os
from collections import defaultdict

import numpy as np


def mcnemar_exact(b, c):
    """Двусторонний точный тест Макнемара. b и c — дискордантные пары.

    Хи-квадрат приближение здесь плохо: при 200 эпизодах дискордантных пар
    бывает десяток, и приближение завышает значимость.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def cluster_bootstrap(by_task, n_boot=20000, seed=0):
    """Бутстрап ПО ЗАДАЧАМ: ресэмплируются задачи целиком, вместе со всеми
    своими эпизодами. Так учитывается, что эпизоды внутри задачи зависимы."""
    tasks = sorted(by_task)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(tasks), len(tasks))
        d = np.concatenate([by_task[tasks[j]] for j in pick])
        out[b] = d.mean()
    return out


def selftest():
    # 1. Макнемар: симметричные дискорданты не значимы, односторонние значимы.
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) > 0.9
    assert mcnemar_exact(0, 10) < 0.01, mcnemar_exact(0, 10)
    assert abs(mcnemar_exact(0, 1) - 1.0) < 1e-12, "одна пара не улика"

    # 2. Кластерный бутстрап ОБЯЗАН давать более широкий интервал, чем наивный,
    #    когда разность различается по задачам. Если нет — кластеризация не
    #    работает, и все выводы будут переуверенными.
    by_task = {f"t{i}": np.full(20, 0.5 if i < 5 else -0.5) for i in range(10)}
    flat = np.concatenate(list(by_task.values()))
    cl = cluster_bootstrap(by_task, 4000, seed=0)
    rng = np.random.default_rng(0)
    naive = np.array([rng.choice(flat, len(flat), replace=True).mean()
                      for _ in range(4000)])
    w_cl = np.percentile(cl, 97.5) - np.percentile(cl, 2.5)
    w_nv = np.percentile(naive, 97.5) - np.percentile(naive, 2.5)
    assert w_cl > 1.5 * w_nv, f"кластерный {w_cl:.3f} не шире наивного {w_nv:.3f}"

    # 3. Односторонняя граница строго выше нижней двусторонней (5% против 2.5%).
    x = cluster_bootstrap({f"t{i}": np.zeros(10) + i * 0.01 for i in range(10)},
                          4000, seed=1)
    assert np.percentile(x, 5) >= np.percentile(x, 2.5) - 1e-12

    # 4. Парность: разность считается по СОВПАДАЮЩИМ ключам, а не по средним
    #    двух наборов. Подмена среднего разностью средних — типичная ошибка.
    a = {1: 1, 2: 0, 3: 1}
    b = {1: 0, 2: 0, 3: 1}
    paired = np.mean([a[k] - b[k] for k in a])
    assert paired == 1 / 3 and paired == np.mean(list(a.values())) - np.mean(list(b.values()))
    a2, b2 = {1: 1, 2: 0}, {2: 0, 3: 1}      # ключи пересекаются частично
    common = sorted(set(a2) & set(b2))
    assert common == [2], "непарные эпизоды должны выпадать, а не усредняться"

    # --- метка обязана нести свою политику ---------------------------------
    ok_map = {"hicora_s0": {"hicora"}, "joint12": {"fast"},
              "coarse24": {"coarse24"}, "чужая_метка": {"что_угодно"}}
    assert check_arm_policies(ok_map)
    # ИМЕННО ВОСПРОИЗВЕДЁННЫЙ ОБХОД: вся рука единообразно подменена, смеси
    # нет, отпечатки одинаковы — прежде это проходило.
    for bad, why in ((dict(ok_map, hicora_s0={"fast"}), "hicora_s0 -> fast"),
                     (dict(ok_map, joint12={"hicora"}), "joint12 -> hicora"),
                     (dict(ok_map, coarse24={"fullbar"}), "coarse24 -> fullbar"),
                     (dict(ok_map, hicora_s0={"?"}), "политика не записана"),
                     (dict(ok_map, hicora_s0={"hicora", "fast"}), "смесь")):
        try:
            check_arm_policies(bad)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"обход принят: {why}")
    # мягкий режим только сообщает
    assert check_arm_policies(dict(ok_map, hicora_s0={"fast"}),
                              strict=False) is False

    # --- руки обязаны делить один черновик ---------------------------------
    assert check_shared_joint({"joint12": {"wj"}, "hicora_s0": {"wj"},
                              "hicora_s1": {"wj"}, "coarse24": set()})
    # ИМЕННО СЛУЧАЙ ВОЗОБНОВЛЕНИЯ: старые ячейки одним чекпойнтом, новые
    # другим; внутри каждой метки единообразно, и прежде это проходило.
    try:
        check_shared_joint({"joint12": {"wj"}, "hicora_s0": {"другой"}})
    except SystemExit:
        pass
    else:
        raise AssertionError("руки с разными весами Joint12 приняты")
    # НЕЗАПИСАННАЯ sha больше не пропускает проверку
    for bad_ in ({"joint12": {"wj"}, "hicora_s0": {"?"}, "hicora_s1": {"wj"}},
                 {"joint12": {"wj"}, "hicora_s0": set(), "hicora_s1": {"wj"}},
                 {"joint12": {"wj"}, "hicora_s0": {"a", "b"}}):
        try:
            check_shared_joint(bad_)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"неудостоверенные веса приняты: {bad_}")
    # метки вне карты K-11e не трогаются: агрегатор общий
    assert check_shared_joint({"fast12": {"a"}, "fast12_rstar": {"b"}})

    # --- s0 и s1 различаются ТОЛЬКО сидом ----------------------------------
    base_h = dict(basis_sha1="bs", rho_sha1="rh", res_norm_sha1="rn",
                  hicora_rank=32, hicora_target="coef",
                  hicora_vla_sha1="hv", hicora_seed=0, hicora_sha1="c0")
    assert check_hicora_replication(
        {"hicora_s0": dict(base_h),
         "hicora_s1": dict(base_h, hicora_seed=1, hicora_sha1="c1")})
    for kw, why in ((dict(basis_sha1="иной"), "базис"),
                    (dict(rho_sha1="иной"), "предел"),
                    (dict(res_norm_sha1="иная"), "норма"),
                    (dict(hicora_rank=16), "ранг"),
                    (dict(hicora_target="star"), "мишень")):
        try:
            check_hicora_replication(
                {"hicora_s0": dict(base_h),
                 "hicora_s1": dict(base_h, hicora_seed=1, **kw)})
        except SystemExit:
            pass
        else:
            raise AssertionError(f"расхождение принято: {why}")
    # одинаковый сид — это не репликация
    try:
        check_hicora_replication({"hicora_s0": dict(base_h),
                                  "hicora_s1": dict(base_h)})
    except SystemExit:
        pass
    else:
        raise AssertionError("одинаковые сиды приняты за репликацию")
    # СОВМЕСТНОЕ ОТСУТСТВИЕ ПОЛЕЙ прежде проходило: обе руки без полей
    # считались репликацией.
    try:
        check_hicora_replication({"hicora_s0": {"hicora_seed": 0},
                                  "hicora_s1": {"hicora_seed": 1}})
    except SystemExit:
        pass
    else:
        raise AssertionError("руки без полей приняты за репликацию")
    # сиды обязаны быть именно 0 и 1
    for a, b in ((1, 0), (2, 3)):
        try:
            check_hicora_replication(
                {"hicora_s0": dict(base_h, hicora_seed=a),
                 "hicora_s1": dict(base_h, hicora_seed=b, hicora_sha1="c1")})
        except SystemExit:
            pass
        else:
            raise AssertionError(f"сиды {a}/{b} приняты")

    print("самопроверка пройдена: руки делят один черновик и различаются "
          "только сидом, карта меток отвергает единообразную подмену "
          "руки, не записанную политику и смесь; Макнемар точный, кластерный бутстрап шире "
          f"наивного ({w_cl:.3f} против {w_nv:.3f}), разность парная")


# КАРТА МЕТКА -> ПОЛИТИКА для K-11e. Отпечаток руки ловит СМЕСЬ внутри
# метки, но не ЕДИНООБРАЗНУЮ подмену: если ВСЕ ячейки `hicora_s0` посчитаны
# политикой `fast`, отпечатки одинаковы и агрегатор молчит. Карта закрывает
# именно этот случай.
ARM_POLICY = {"fullbar": "fullbar", "coarse24": "coarse24",
              "joint12": "fast", "hicora_s0": "hicora",
              "hicora_s1": "hicora"}


def check_shared_joint(wsha_by_arm, known=None):
    """Сравниваемые руки K-11e обязаны делить ОДИН черновик.

    Внутри метки единообразие уже проверено, но этого мало. Реальный случай
    при возобновлении: старые 400 ячеек `joint12` посчитаны одним чекпойнтом
    Joint12, новые `hicora_*` — другим. Внутри каждой метки sha единообразна,
    карта политик верна, и агрегатор принимал бы сравнение. Тогда HiCoRA
    сравнивалась бы НЕ СО СВОИМ черновиком.

    Проверяются только метки из карты K-11e: агрегатор общий, и старые
    эксперименты могли законно сравнивать руки с разными весами.
    """
    m = ARM_POLICY if known is None else known
    need = [a for a in m if m[a] in ("fast", "hicora")]
    present = [a for a in need if a in wsha_by_arm]
    if not present:
        return True                      # ни одной руки K-11e — не наш случай
    bad, got = [], {}
    for arm in need:
        shas = {x for x in wsha_by_arm.get(arm, set()) if x and x != "?"}
        if arm in wsha_by_arm and not shas:
            bad.append(f"{arm}: sha весов не записана")
        elif len(shas) > 1:
            bad.append(f"{arm}: sha весов несколько {sorted(shas)}")
        elif shas:
            got[arm] = next(iter(shas))
    if bad:
        raise SystemExit("ВЕСА Joint12 НЕ УДОСТОВЕРЕНЫ:\n    "
                         + "\n    ".join(bad))
    if len(set(got.values())) > 1:
        raise SystemExit(
            f"РУКИ ДЕЛЯТ РАЗНЫЕ ВЕСА Joint12: {got}. HiCoRA сравнивалась бы "
            f"не со своим черновиком — сравнение недействительно.")
    return True


def check_hicora_replication(hic_by_arm):
    """`hicora_s0` и `hicora_s1` обязаны различаться ТОЛЬКО сидом.

    Иначе это не репликация по сиду, а две разные конфигурации, и требование
    «выполнить на обоих» ничего не удостоверяет.
    """
    arms = sorted(a for a in hic_by_arm if a.startswith("hicora_"))
    if len(arms) < 2:
        return True
    same = ("basis_sha1", "rho_sha1", "res_norm_sha1", "hicora_rank",
            "hicora_target", "hicora_vla_sha1")
    bad = []
    for fld in same:
        vals = {a: hic_by_arm[a].get(fld) for a in arms}
        # СОВМЕСТНОЕ ОТСУТСТВИЕ ПОЛЯ — ТОЖЕ ОТКАЗ. Прежде проверялось лишь
        # несовпадение, и две руки без полей вовсе считались репликацией.
        if any(v is None for v in vals.values()):
            bad.append(f"{fld}: отсутствует у {[a for a in arms if hic_by_arm[a].get(fld) is None]}")
        elif len({str(v) for v in vals.values()}) > 1:
            bad.append(f"{fld}: {vals}")
    # СИДЫ ОБЯЗАНЫ БЫТЬ ИМЕННО 0 И 1, а не просто разными.
    for a, want in (("hicora_s0", 0), ("hicora_s1", 1)):
        if a not in hic_by_arm:
            continue
        got = hic_by_arm[a].get("hicora_seed")
        if got is None:
            bad.append(f"{a}: сид не записан")
        elif int(got) != want:
            bad.append(f"{a}: сид {got} вместо {want}")
    if bad:
        raise SystemExit(
            "РУКИ hicora_s0 И hicora_s1 РАЗЛИЧАЮТСЯ НЕ ТОЛЬКО СИДОМ:\n    "
            + "\n    ".join(bad)
            + "\n  Тогда это не репликация, и «выполнить на обоих» ничего не "
              "удостоверяет.")
    return True


def check_arm_policies(pol_by_arm, expect=None, strict=True):
    """Каждая известная метка обязана нести свою политику.

    `strict` требует, чтобы у метки была ровно одна политика и чтобы она
    совпала с ожидаемой. Метки вне карты пропускаются: агрегатор общий, и
    старые эксперименты про K-11e ничего не знают.
    """
    m = ARM_POLICY if expect is None else expect
    bad = []
    for arm, pols in sorted(pol_by_arm.items()):
        if arm not in m:
            continue
        if len(pols) != 1:
            bad.append(f"{arm}: политик несколько {sorted(pols)}")
            continue
        got = next(iter(pols))
        if got == "?":
            bad.append(f"{arm}: политика не записана в ячейках")
        elif got != m[arm]:
            bad.append(f"{arm}: политика {got}, ожидалась {m[arm]}")
    if bad and strict:
        raise SystemExit(
            "МЕТКА НЕ СООТВЕТСТВУЕТ ПОЛИТИКЕ:\n    " + "\n    ".join(bad)
            + "\n  Единообразно подменённая рука дала бы одинаковые "
              "отпечатки и прошла бы проверку на смесь.")
    return not bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--glob", default="data/k6h/*.json")
    ap.add_argument("--margin", type=float, default=10.0,
                    help="граница не-хуже-чем, в пунктах успеха")
    # ПОЛЕ РАЗДЕЛЕНИЯ РУК. По умолчанию — levels 1 против 3, то есть K-6h без
    # изменений. K-9d кладёт в файлы поле arm, и та же статистика применяется к
    # паре fast12/coarse24. Обобщается ТОЛЬКО ключ; тесты, бутстрап и правило
    # чтения не трогаются, иначе пришлось бы заново подтверждать оценщик.
    ap.add_argument("--field", default="levels",
                    help="поле файла, различающее руки (levels или arm)")
    ap.add_argument("--test", default=None,
                    help="значение --field у ИСПЫТУЕМОЙ руки (по умолчанию 1)")
    ap.add_argument("--ref", default=None,
                    help="значение --field у ОПОРНОЙ руки (по умолчанию 3)")
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-hash-mismatch", action="store_true",
                    help="НЕ используйте: расхождение хешей означает, что "
                         "эпизоды стартовали из разных состояний")
    # НЕПАРНЫЕ ЯЧЕЙКИ — ОТКАЗ, А НЕ СТРОЧКА В ОТЧЁТЕ. Ячейка пропадает, когда
    # процесс упал или не дошёл; падать чаще может именно испытуемая рука, и
    # тогда из выборки систематически исчезают её худшие эпизоды. Прежняя
    # версия печатала «НЕПАРНЫХ n» и всё равно выдавала вердикт.
    ap.add_argument("--allow-unpaired", action="store_true",
                    help="считать по неполной развёртке. Вердикт при этом "
                         "недействителен: потеря ячеек не случайна.")
    # ОЖИДАЕМЫЙ РАЗМЕР ЗАДАЁТСЯ ЗАРАНЕЕ. «Столько пар, сколько нашлось» —
    # это молчаливое согласие на любую развёртку, включая ту, где половина
    # ячеек не запускалась вовсе. Непарность ловит пропажу ОДНОЙ руки, а
    # пропажу ЦЕЛОЙ ячейки — только сверка с ожидаемым числом.
    ap.add_argument("--expect-pairs", type=int, default=None)
    ap.add_argument("--expect-tasks", type=int, default=None)
    ap.add_argument("--require-full-hash", action="store_true",
                    help="требовать init_hash_full у ОБЕИХ рук каждой пары: "
                         "он включает камеру запястья, в которую политика "
                         "тоже смотрит")
    ap.add_argument("--allow-extra-arms", action="store_true",
                    help="разрешить в ячейке метки помимо --test и --ref. "
                         "НУЖЕН при намеренно многоруком эксперименте: три "
                         "руки разбираются тремя попарными вызовами, и в "
                         "каждом третья метка законно лишняя. Без флага "
                         "посторонняя метка — признак чужого прогона.")
    # PROVENANCE FAIL-CLOSED. Проверки, работающие «если поле есть», открыты
    # настежь: ячейки старой версии без rollout_seed проходят их молча, а
    # раннер пропускает готовые JSON — так частично выполненный прежний прогон
    # смешивается с новым незаметно. Разные script_sha1 по той же причине
    # больше не предупреждение.
    ap.add_argument("--allow-legacy-provenance", action="store_true",
                    help="читать ячейки без rollout_seed и смешивать версии "
                         "скрипта. Гарантии при этом слабее заявленных; "
                         "нужен для файлов, снятых до введения поля (K-9d).")
    ap.add_argument("--allow-arm-policy-mismatch", action="store_true",
                    help="разрешить метке нести не ту политику. Нужен только "
                         "для разбора старых каталогов")
    ap.add_argument("--hypothesis", choices=["noninferiority", "superiority"],
                    default="noninferiority",
                    help="какая гипотеза проверяется. При superiority допуск "
                         "обязан быть нулевым, а вердикт называется "
                         "превосходством, а не не-худшестью")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    selftest()
    if args.hypothesis == "superiority" and abs(args.margin) > 1e-12:
        raise SystemExit(
            f"--hypothesis superiority требует --margin 0, задано "
            f"{args.margin}: превосходство проверяется границей у нуля")
    if args.selftest:
        return

    if args.field == "levels":
        A = int(args.test) if args.test is not None else 1
        B = int(args.ref) if args.ref is not None else 3
    else:
        if args.test is None or args.ref is None:
            raise SystemExit(f"при --field {args.field} нужны --test и --ref")
        A, B = args.test, args.ref
    if A == B:
        raise SystemExit("--test и --ref обязаны различаться")

    files = sorted(globmod.glob(args.glob))
    if not files:
        raise SystemExit(f"нет файлов по {args.glob}")
    cells = defaultdict(dict)     # (tag,suite,task,ens,H,init_id) -> {рука: ep}
    shas, ckpts = set(), set()
    # SHA ВЕСОВ ПРОВЕРЯЕТСЯ ПО КАЖДОЙ РУКЕ ОТДЕЛЬНО, а не глобально. Разные
    # веса у РАЗНЫХ меток — это и есть смысл многорукого сравнения
    # (frozen12_rstar и joint12 обязаны отличаться). Ошибка — два разных
    # набора весов у ОДНОЙ метки: тогда часть её ячеек посчитана другой сетью.
    # Первая версия этой проверки была глобальной и падала на законных данных.
    wsha_by_arm = defaultdict(set)
    fp_by_arm, pol_by_arm = defaultdict(set), defaultdict(set)
    hic_by_arm = {}
    vlasha = set()
    for f in files:
        d = json.load(open(f))
        shas.add(d.get("script_sha1", "?")); ckpts.add(d.get("ckpt", "?"))
        if args.field not in d:
            raise SystemExit(
                f"в {f} нет поля «{args.field}» — файл получен другим "
                f"скриптом. Проверьте --glob и --field.")
        # Путь к чекпойнту не удостоверяет ничего: best_imitation.pt
        # перезаписывается каждой лучшей эпохой. Удостоверяет sha весов.
        # ПОЛИТИКА ЗАПИСЫВАЕТСЯ БЕЗУСЛОВНО. Прежде она собиралась только
        # внутри `if isinstance(joint, dict)`, поэтому ячейки fullbar и
        # coarse24 в проверку не попадали, и метка могла содержать смесь
        # coarse24 с hicora незамеченно.
        arm_ = str(d[args.field])
        pol_ = str(d.get("policy", "?"))
        pol_by_arm[arm_].add(pol_)
        j = d.get("joint")
        if isinstance(j, dict):
            wsha_by_arm[arm_].add(j.get("weights_sha1", "?"))
            vlasha.add(j.get("joint12_vla_sha1", "?"))
        # ЕДИНЫЙ ОТПЕЧАТОК РУКИ. Для HiCoRA веса Joint12 одинаковы у
        # `joint12` и у `hicora_*`, поэтому sha весов руки не различает:
        # половина ячеек могла быть посчитана ДРУГОЙ головой, с другим
        # базисом, пределом, нормой или сидом — и агрегатор молчал бы.
        #
        # ОТСУТСТВИЕ ОТПЕЧАТКА У HICORA — ОТКАЗ, а не пропуск: иначе защиту
        # снимало бы простое его отсутствие во всех ячейках.
        if pol_ == "hicora":
            if not isinstance(j, dict):
                raise SystemExit(
                    f"{os.path.basename(f)}: политика hicora без словаря "
                    f"происхождения")
            fp = j.get("arm_fingerprint")
            if not fp:
                raise SystemExit(
                    f"{os.path.basename(f)}: у руки hicora нет "
                    f"arm_fingerprint — какая именно голова считала эту "
                    f"ячейку, не доказуемо")
            fp_by_arm[arm_].add(str(fp))
            hic_by_arm[arm_] = {k_: j.get(k_) for k_ in (
                "basis_sha1", "rho_sha1", "res_norm_sha1", "hicora_rank",
                "hicora_target", "hicora_vla_sha1", "hicora_seed",
                "hicora_sha1")}
        elif isinstance(j, dict) and j.get("arm_fingerprint"):
            fp_by_arm[arm_].add(str(j["arm_fingerprint"]))
        # run_tag В КЛЮЧЕ: ячейки K-6h и K-9d могут лежать рядом и совпадать по
        # (suite, task, ens, H, init_id). Без тега они молча склеились бы в
        # одну пару, и сравнивались бы эпизоды из разных экспериментов.
        for e in d["episodes"]:
            key = (d.get("run_tag", "k6h"), d["suite"], d["task_id"],
                   d.get("ensemble", "?"), d["horizon"], e["init_state_id"])
            lv = d[args.field]
            if lv in cells[key]:
                raise SystemExit(
                    f"дубль ячейки {key}, {args.field}={lv}: два файла "
                    f"описывают один эпизод. Проверьте, не запущен ли один "
                    f"блок дважды.")
            cells[key][lv] = e
    if len(shas) > 1:
        if not args.allow_legacy_provenance:
            raise SystemExit(
                f"файлы получены РАЗНЫМИ версиями скрипта: {sorted(shas)}.\n"
                f"Раннер пропускает готовые JSON, поэтому так незаметно "
                f"смешивается частично выполненный прежний прогон с новым.\n"
                f"Считайте в чистом каталоге под новым --run-tag или, понимая "
                f"последствия, --allow-legacy-provenance.")
        print(f"  ВНИМАНИЕ: файлы получены РАЗНЫМИ версиями скрипта: {shas}")
    if len(ckpts) > 1:
        raise SystemExit(f"разные чекпойнты в одном сравнении: {ckpts}")
    check_arm_policies(pol_by_arm, strict=not args.allow_arm_policy_mismatch)
    if not args.allow_arm_policy_mismatch:
        check_shared_joint(wsha_by_arm)
        check_hicora_replication(hic_by_arm)
    for a_ in sorted(pol_by_arm):
        if a_ in ARM_POLICY:
            print(f"  метка {a_}: политика {sorted(pol_by_arm[a_])[0]} "
                  f"(ожидалась {ARM_POLICY[a_]})")
    mixed_fp = {a: s for a, s in fp_by_arm.items() if len(s) > 1}
    if mixed_fp:
        raise SystemExit(
            f"ВНУТРИ ОДНОЙ МЕТКИ РАЗНЫЕ ОТПЕЧАТКИ РУКИ: {mixed_fp}. Часть "
            f"ячеек посчитана другой моделью — другой головой, базисом, "
            f"пределом, нормой или сидом. Сравнение недействительно.")
    mixed_pol = {a: s for a, s in pol_by_arm.items() if len(s) > 1}
    if mixed_pol:
        raise SystemExit(
            f"ВНУТРИ ОДНОЙ МЕТКИ РАЗНЫЕ ПОЛИТИКИ: {mixed_pol}. Метка руки "
            f"обязана соответствовать одной политике.")
    for a in sorted(fp_by_arm):
        if fp_by_arm[a]:
            print(f"  рука {a}: отпечаток {sorted(fp_by_arm[a])[0]}")
    mixed = {a: s for a, s in wsha_by_arm.items() if len(s) > 1}
    if mixed:
        raise SystemExit(
            f"у одной и той же руки смешаны РАЗНЫЕ веса: "
            f"{ {a: sorted(s) for a, s in mixed.items()} }.\nЧасть её ячеек "
            f"посчитана другой сетью. Развёртку надо вести в каталоге, "
            f"привязанном к sha весов.")
    # ВЕРСИЯ МОДУЛЯ ИНФЕРЕНСА обязана быть одна у всех рук: forward_joint_fast
    # определяет исполняемую сеть не меньше, чем веса.
    if len(vlasha) > 1:
        raise SystemExit(
            f"руки исполнены разными версиями joint12_vla.py: "
            f"{sorted(vlasha)}. Сравнение недействительно.")
    for a in sorted(wsha_by_arm):
        print(f"  рука {a}: веса sha {sorted(wsha_by_arm[a])[0]}")
    if vlasha:
        print(f"  joint12_vla sha {sorted(vlasha)[0]}")

    print(f"  файлов {len(files)}, ячеек {len(cells)}; "
          f"{args.field}: испытуемая {A}, опора {B}")
    res = {}
    for run in sorted({k[0] for k in cells}):
     for ens in sorted({k[3] for k in cells if k[0] == run}):
      for H in sorted({k[4] for k in cells
                       if k[0] == run and k[3] == ens}):
        sub = [k for k in cells if k[0] == run and k[3] == ens and k[4] == H]
        keys = [k for k in sub if A in cells[k] and B in cells[k]]
        # НЕПАРНОСТЬ ОПРЕДЕЛЯЕТСЯ ПО НАЛИЧИЮ НУЖНЫХ МЕТОК, а не по числу
        # записей: ячейка с тремя посторонними метками и без A имеет длину 3 и
        # прошла бы проверку `len(cells[k]) < 2`, хотя пары в ней нет.
        unpaired = [k for k in sub if not (A in cells[k] and B in cells[k])]
        extra = {lab for k in sub for lab in cells[k] if lab not in (A, B)}
        if extra and not args.allow_extra_arms:
            raise SystemExit(
                f"{run} ens={ens} H={H}: в ячейках есть посторонние метки "
                f"{sorted(extra)} помимо {A} и {B}.\nВ сравнение попал чужой "
                f"прогон — уточните --glob или --field/--test/--ref.")
        if not keys:
            continue
        # ПОЛНЫЙ ХЕШ ПРОВЕРЯЕТСЯ ТАМ, ГДЕ ОН ЕСТЬ У ОБЕИХ РУК. Он включает
        # камеру на запястье, в которую политика тоже смотрит; совпадение
        # только по agentview — более слабое условие, чем требуется.
        bad = [k for k in keys
               if cells[k][A].get("init_hash") != cells[k][B].get("init_hash")]
        badf = [k for k in keys
                if cells[k][A].get("init_hash_full") is not None
                and cells[k][B].get("init_hash_full") is not None
                and (cells[k][A]["init_hash_full"]
                     != cells[k][B]["init_hash_full"])]
        if (bad or badf) and not args.allow_hash_mismatch:
            raise SystemExit(
                f"{run} ens={ens} H={H}: у {len(bad)} из {len(keys)} пар "
                f"РАЗНЫЕ хеши начального наблюдения ({len(badf)} по полному "
                f"хешу с камерой запястья), например {(bad or badf)[0]}.\n"
                f"Это значит, что эпизоды с одним init_state_id стартовали из "
                f"разных состояний и парное сравнение недействительно.")
        n_full = sum(1 for k in keys
                     if cells[k][A].get("init_hash_full") is not None
                     and cells[k][B].get("init_hash_full") is not None)
        # СИД РАСКАТКИ У ПАРЫ ОБЯЗАН СОВПАДАТЬ. Режим `block` в гейте даёт
        # seed + 1000 * init_start, поэтому руки с разным числом сред получают
        # РАЗНЫЕ сиды на одном init_state_id, и сравнение мерило бы размер
        # батча вместе с сидом. Для таких пар в гейте есть режим `fixed`;
        # здесь проверяется, что им действительно воспользовались.
        noseed = [k for k in keys
                  if cells[k][A].get("rollout_seed") is None
                  or cells[k][B].get("rollout_seed") is None]
        if noseed and not args.allow_legacy_provenance:
            raise SystemExit(
                f"{run} ens={ens} H={H}: у {len(noseed)} из {len(keys)} пар "
                f"нет rollout_seed хотя бы у одной руки, например "
                f"{noseed[0]}.\nСверить сиды невозможно, а именно они отличают "
                f"разницу рук от разницы раскатки. Пересчитайте эти ячейки "
                f"или, понимая последствия, --allow-legacy-provenance.")
        badseed = [k for k in keys
                   if cells[k][A].get("rollout_seed") is not None
                   and cells[k][B].get("rollout_seed") is not None
                   and cells[k][A]["rollout_seed"] != cells[k][B]["rollout_seed"]]
        if badseed:
            raise SystemExit(
                f"{run} ens={ens} H={H}: у {len(badseed)} из {len(keys)} пар "
                f"РАЗНЫЕ сиды раскатки, например {badseed[0]}:\n"
                f"  {A}: {cells[badseed[0]][A]['rollout_seed']}, "
                f"{B}: {cells[badseed[0]][B]['rollout_seed']}.\n"
                f"Сравнение измерило бы разницу рук вместе с разницей сида. "
                f"Перезапустите с --rollout-seed-mode fixed у ОБЕИХ рук.")
        if args.require_full_hash and n_full < len(keys):
            raise SystemExit(
                f"{run} ens={ens} H={H}: полный хеш есть только у {n_full} из "
                f"{len(keys)} пар. Часть ячеек снята версией без камеры "
                f"запястья, и парность у них проверена слабее требуемого.")
        if unpaired and not args.allow_unpaired:
            raise SystemExit(
                f"{run} ens={ens} H={H}: {len(unpaired)} ячеек без пары при "
                f"{len(keys)} полных. Развёртка неполная, и вердикт по ней\n"
                f"недействителен: падать чаще может именно испытуемая рука, и "
                f"тогда из выборки систематически исчезают её худшие эпизоды.\n"
                f"Досчитайте недостающие ячейки или, понимая последствия, "
                f"--allow-unpaired.")

        if args.expect_pairs is not None and len(keys) != args.expect_pairs:
            raise SystemExit(
                f"{run} ens={ens} H={H}: {len(keys)} полных пар, ожидалось "
                f"{args.expect_pairs}. Развёртка не того размера, который "
                f"планировался.")
        n_tasks_here = len({k[2] for k in keys})
        if args.expect_tasks is not None and n_tasks_here != args.expect_tasks:
            raise SystemExit(
                f"{run} ens={ens} H={H}: {n_tasks_here} задач, ожидалось "
                f"{args.expect_tasks}. Кластерный бутстрап по задачам на "
                f"неполном наборе занижает ширину интервала.")

        by_task = defaultdict(list)
        b = c = 0
        for k in keys:
            s1 = int(cells[k][A]["success"]); s3 = int(cells[k][B]["success"])
            by_task[k[2]].append(s1 - s3)
            b += (s1 == 1 and s3 == 0); c += (s1 == 0 and s3 == 1)
        by_task = {t: np.asarray(v, float) for t, v in by_task.items()}
        flat = np.concatenate(list(by_task.values()))
        micro = flat.mean() * 100
        macro = float(np.mean([v.mean() for v in by_task.values()])) * 100
        boot = cluster_bootstrap(by_task, args.n_boot, args.seed) * 100
        lo1 = float(np.percentile(boot, 5))
        hi1 = float(np.percentile(boot, 95))
        lo2, hi2 = (float(np.percentile(boot, 2.5)),
                    float(np.percentile(boot, 97.5)))
        p = mcnemar_exact(b, c)
        r1 = np.mean([int(cells[k][A]["success"]) for k in keys]) * 100
        r3 = np.mean([int(cells[k][B]["success"]) for k in keys]) * 100

        tag = f"{run}, ens={ens}, H={H}"
        print(f"\n{'=' * 68}\n  {tag}: {len(keys)} пар, "
              f"{len(by_task)} задач" + (f", НЕПАРНЫХ {len(unpaired)}"
                                         if unpaired else ""))
        print(f"    успех {args.field}={A}: {r1:5.1f}%     "
              f"{args.field}={B}: {r3:5.1f}%")
        print(f"    парная разность ({A} минус {B}): micro {micro:+.1f} пп, "
              f"macro {macro:+.1f} пп")
        # ДОЛЯ ДИСКОРДАНТНЫХ ПАР — главное число для контроля собственного
        # шума. У пары «глубина 12 против 24» она составила 58/400 = 14.5%.
        # Если тот же показатель у пары «одна и та же политика при другой
        # форме батча» окажется сопоставимым, различие рук объясняется
        # хаотическим расхождением траекторий, а не глубиной.
        disc = (b + c) / max(len(keys), 1)
        print(f"    дискордантных пар: {A} лучше {b}, {B} лучше {c}, "
              f"всего {b + c} = {disc:.1%}; Макнемар p = {p:.3f}")
        # Оракул постфактум: он знает исход и потому не доказывает, что
        # выбор предсказуем заранее. Печатается как верхняя граница запаса.
        both_fail = sum(1 for k in keys if not cells[k][A]["success"]
                        and not cells[k][B]["success"])
        print(f"    оракул постфактум {100 * (1 - both_fail / len(keys)):.1f}% "
              f"(обе провалились в {both_fail}) — ЗНАЕТ ИСХОД, запасом "
              f"адаптивности НЕ является")
        print(f"    кластерный бутстрап по задачам: "
              f"95% ДИ [{lo2:+.1f}, {hi2:+.1f}] пп"
              + (f", полный хеш сверен у {n_full}" if n_full else ""))
        print(f"    односторонние границы (5%/95%): "
              f"нижняя {lo1:+.1f}, верхняя {hi1:+.1f} пп")
        # ТРИ ИСХОДА, А НЕ ДВА. Не-худшесть доказывает нижняя граница выше
        # -margin; ухудшение более чем на margin доказывает ВЕРХНЯЯ граница
        # ниже -margin. Между ними не доказано ничего, и читать «нижняя ниже
        # порога» как доказанное ухудшение — та же ошибка, что читать
        # неотвергнутую нулевую гипотезу как доказанное равенство.
        non_inf = bool(lo1 > -args.margin)
        inferior = bool(hi1 < -args.margin)
        verdict = ("НЕ ХУЖЕ (доказано)" if non_inf else
                   "ХУЖЕ более чем на границу (доказано)" if inferior else
                   "НЕ ДОКАЗАНО НИЧЕГО")
        print(f"    при границе {args.margin:.0f} пп: {verdict}")
        if not non_inf and not inferior:
            print(f"    ни одна из двух односторонних гипотез не подтверждена; "
                  f"точечная оценка {micro:+.1f} пп сама по себе НЕ вывод")
        # ГИПОТЕЗА НАЗЫВАЕТСЯ СВОИМ ИМЕНЕМ. При нулевом допуске математика
        # односторонней границы та же, но вердикт «НЕ ХУЖЕ» вводил бы в
        # заблуждение: проверяется ПРЕВОСХОДСТВО, и в JSON должно стоять оно.
        superior = bool(args.hypothesis == "superiority" and lo1 > 0)
        if args.hypothesis == "superiority":
            print(f"    ГИПОТЕЗА ПРЕВОСХОДСТВА: нижняя односторонняя граница "
                  f"{lo1:+.1f} пп — " + ("ПРЕВОСХОДСТВО ДОКАЗАНО"
                                         if superior else "НЕ ДОКАЗАНО"))
        res[tag] = dict(hypothesis=args.hypothesis, superior=superior,
                        n_pairs=len(keys), n_tasks=len(by_task),
                        n_unpaired=len(unpaired), n_full_hash=n_full,
                        field=args.field, arm_test=str(A), arm_ref=str(B),
                        rate_l1=r1, rate_l3=r3,
                        micro=micro, macro=macro, disc_1=b, disc_3=c,
                        mcnemar_p=p, ci95=[lo2, hi2], discordant_frac=disc,
                        both_fail=both_fail,
                        hindsight_oracle=1 - both_fail / len(keys),
                        lower_1s=lo1, upper_1s=hi1,
                        margin=args.margin, non_inferior=non_inf,
                        inferior=inferior,
                        undetermined=bool(not non_inf and not inferior))

    if not res:
        raise SystemExit(
            f"не нашлось ни одной пары {args.field}={A} / {args.field}={B}")
    print(f"\n  ЧИТАТЬ по односторонней границе, правило записано в докстроке")
    print(f"  k6h_coarse_gate.py ДО запуска. «ДИ пересекает ноль» — не вывод.")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                    exist_ok=True)
        json.dump(dict(cells=res, files=files, script_shas=sorted(shas)),
                  open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"  сохранено: {args.out}")


if __name__ == "__main__":
    main()
