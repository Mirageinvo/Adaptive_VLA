"""K-11h: стоимость ИМЕННО ТОЙ политики, которая проверена в K-11e.

ЧЕЙ ЭТО СЧЁТ. HiCoRA, joint12 и fullbar исполняются как в K-11e. coarse24
меряется в ОПТИМИЗИРОВАННОЙ однопроходной реализации, выдающей то же грубое
действие: в гейте она шла через все три блока generate() и выбрасывала уровни
1-2, и её тамошнее время не было бы честным comparator. То есть порог 1.10
сравнивает HiCoRA с ИДЕАЛИЗИРОВАННЫМ coarse24, что строже для нас.

ЧТО МЕРЯЕТСЯ (K-11h-A, зарегистрированный критерий, батч 1):
  fullbar    — 24 слоя, ТРИ прохода, сборка из трёх уровней;
  coarse24   — 24 слоя, ОДИН проход, сборка из уровня 0;
  joint12    — 12 слоёв, один проход, веса Joint-12;
  hicora_s0  — 24 слоя, один проход, q0 со слоя 12 + поправка от h24;
  hicora_s1  — то же, вторая голова.

    max_s  T(hicora_s) / T(coarse24)  <= 1.10
    min_s  T(fullbar)  / T(hicora_s)  >= 1.80

ЧТО ЭТО НЕ ЕСТЬ. Потоковая политика («исполнить черновик со слоя 12, пока
считаются слои 13-24») здесь НЕ мерится: её нет в коде. forward_hicora
исполняет все 24 слоя и лишь затем берёт q0 из сохранённого h12. У потоковой
руки было бы ДВА декодирования, ГРУБОЕ первое действие и СВОЙ, неизвестный
успех. Соединять 92.0% из K-11e с задержкой в 12 слоёв нельзя.

=== DTYPE: КОНФАУНДЕР, ИЗ-ЗА КОТОРОГО ПЕРВАЯ ВЕРСИЯ БЫЛА НЕГОДНОЙ ===========
Прежняя версия приводила веса Joint12 к fp32 (так делает `weights()` в K-9i,
где сравнение шло ВНУТРИ группы) и оставляла fullbar/coarse24 на fp16. Тогда
joint12 и HiCoRA несли лишнюю двухбайтовую копию примерно 880 млн обучаемых
параметров — около 1.68 ГиБ, что и составляло всю наблюдённую разницу памяти
(9961 - 8245 = 1716 МиБ). Объяснения «+21% из-за init_joint_fast» и «головы
дороже сэкономленной авторегрессии» были НЕВЕРНЫ: измерение смешивало
архитектуру с хранением fp32.

Отсюда ДВА РЕЖИМА, отвечающих на разные вопросы:
  as-executed — веса черновика в fp32 под autocast fp16, РОВНО как в гейте
                K-11e. Стоимость фактически исполнявшейся реализации.
                Зарегистрированный режим: сравнивать надо то, что считало
                успех. Память при этом НЕ является архитектурной.
  uniform-trunk — единый inference-dtype У СТВОЛА. Название именно такое:
                голова поправки, кодовые книги и декодер остаются fp32, так
                что «все веса в одном dtype» было бы неправдой. Отвечает про
                АРХИТЕКТУРУ, но это другая реализация, и K-11e её не исполнял;
                прогон маркируется как НЕ зарегистрированный.

ПАМЯТЬ МЕРИТСЯ ОТДЕЛЬНЫМ ПРОЦЕССОМ НА РУКУ. В одном процессе состояния всех
рук предзагружены на карту, и `max_memory_allocated` не принадлежит ни одной
из них. Режим `--mem-only ARM` собирает ТОЛЬКО нужное этой руке (fullbar и
coarse24 вообще не вызывают init_joint_fast) и печатает пик.

ПОРЯДОК РУК БАЛАНСИРУЕТСЯ ЦИКЛИЧЕСКИМИ СДВИГАМИ. Запас до порога 1.10
составляет около 1.5%, а фиксированный порядок даёт систематический эффект
позиции: первая рука цикла стабильно в иных условиях по прогреву и частоте,
чем последняя. При сдвиге каждая рука стоит в каждой позиции ровно reps/5 раз.
Порядок каждого повтора сохраняется в JSON.

СЫРЫЕ ВРЕМЕНА СОХРАНЯЮТСЯ ЦЕЛИКОМ. Прежняя версия писала только
median/mean/p95, и «пересчитать медианы по JSON» было неверно: пересчитать
можно было лишь отношения уже записанных медиан. Теперь в JSON лежат все
парные замеры, и рядом печатается ДИ ПАРНОГО отношения как проверка
устойчивости. Зарегистрированная точечная оценка остаётся отношением медиан и
задним числом не меняется.

ПОЧЕМУ ПОРОГ 1.10, А НЕ 0.60. Прежнее предложение «0.60 от coarse24»
противоречило K-9i: 12 слоёв против 24 дают 65.1 против 84.7 мс, то есть 0.77
на батче 1 и 0.88 на батче 10. Постоянная часть — башня зрения, подготовка
входа, декодер — с числом слоёв не масштабируется.

Запуск:
    python experiments/k11h_latency.py --selftest
    python experiments/k11h_latency.py --ckpt <hf> --joint12 data/k9d_ep3.pt \\
        --hicora-s0 ... --hicora-s1 ... --out data/k11h/latency.json
"""

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

N_POS, N_LEVEL = 16, 3
CONFIGS = ("fullbar", "coarse24", "joint12", "hicora_s0", "hicora_s1")
HICORA = ("hicora_s0", "hicora_s1")
BASE_W = ("fullbar", "coarse24")          # идут на исходных весах
NEEDS_JOINT = ("joint12",) + HICORA
MAX_VS_COARSE = 1.10
MIN_VS_FULLBAR = 1.80
PRIMARY_BATCH = 1
# ЗАРЕГИСТРИРОВАННЫЙ ПРОТОКОЛ. Отклонение по любому полю делает прогон НЕ
# зарегистрированным, и это пишется в JSON и в финальную печать. Прежде
# `--batches 1 --reps 1 --warmup 0` выдавал «K-11h-A ПРОЙДЕН» без оговорок.
REGISTERED = dict(batches=[1, 10], reps=200, warmup=20, dtype="float16",
                  pos_offset=4, weight_mode="as-executed",
                  cache="data/k9_teacher_150k.npz",
                  max_vs_coarse=MAX_VS_COARSE,
                  min_vs_fullbar=MIN_VS_FULLBAR)
PASSES = {"fullbar": N_LEVEL, "coarse24": 1, "joint12": 1,
          "hicora_s0": 1, "hicora_s1": 1}
LAYERS = {"fullbar": 24, "coarse24": 24, "joint12": 12,
          "hicora_s0": 24, "hicora_s1": 24}
DECODES = {c: 1 for c in CONFIGS}


def stats(xs):
    a = np.asarray(xs, float)
    return dict(median=float(np.median(a)), mean=float(a.mean()),
                p95=float(np.percentile(a, 95)), std=float(a.std()),
                n=int(a.size))


def rotations(arms, reps):
    """Сбалансированные циклические сдвиги: каждая рука в каждой позиции.

    ЗАЧЕМ. Фиксированный порядок оставляет систематический эффект позиции:
    первая рука цикла всегда в иных условиях по прогреву и частоте, чем
    последняя. Запас до порога около 1.5%, то есть эффект позиции сравним с
    измеряемым. Случайная перестановка балансирует лишь в среднем; сдвиг даёт
    ТОЧНЫЙ баланс, когда reps делится на число рук.
    """
    n = len(arms)
    if reps % n:
        raise SystemExit(
            f"reps={reps} не делится на число рук {n}: порядок нельзя "
            f"сбалансировать точно, и эффект позиции достанется кому-то "
            f"чаще. Возьмите reps, кратное {n}")
    return [tuple(arms[(i + r) % n] for i in range(n)) for r in range(reps)]


def check_balance(orders, arms):
    """Каждая рука обязана постоять в каждой позиции одинаково часто."""
    n = len(arms)
    cnt = {(a, p): 0 for a in arms for p in range(n)}
    for o in orders:
        if sorted(o) != sorted(arms):
            raise SystemExit(f"в цикле не все руки: {o}")
        for p, a in enumerate(o):
            cnt[(a, p)] += 1
    vals = set(cnt.values())
    if len(vals) != 1:
        raise SystemExit(f"порядок не сбалансирован: частоты {sorted(vals)}")
    return vals.pop()


def check_joint_state(j_state, base_state):
    """Набор весов черновика обязан СОВПАДАТЬ с исходным, а не входить в него.

    ПРОВЕРКА В ОБЕ СТОРОНЫ. Прежде искались только ключи Joint12, которых нет
    среди исходных. Обратный случай — ключ есть в модели, но ОТСУТСТВУЕТ в
    чекпойнте — проходил молча, и рука joint12 исполнялась бы на смеси
    обученных и исходных весов.
    """
    bad = []
    lost = sorted(k for k in j_state if k not in base_state)
    if lost:
        bad.append(f"{len(lost)} весов Joint12 не сохранены как исходные: "
                   f"{lost[:4]} — тогда fullbar пошёл бы на смеси")
    missing = sorted(k for k in base_state if k not in j_state)
    if missing:
        bad.append(f"{len(missing)} обучаемых весов НЕТ в чекпойнте: "
                   f"{missing[:4]} — joint12 пошёл бы на смеси обученных и "
                   f"исходных")
    for k in sorted(set(j_state) & set(base_state)):
        if tuple(j_state[k].shape) != tuple(base_state[k].shape):
            bad.append(f"{k}: форма {tuple(j_state[k].shape)} против "
                       f"{tuple(base_state[k].shape)}")
    if bad:
        raise SystemExit("ЧЕКПОЙНТ JOINT12 НЕ СООТВЕТСТВУЕТ МОДЕЛИ:\n    "
                         + "\n    ".join(bad))
    return True


def read_gate(med, max_vs_coarse=MAX_VS_COARSE,
              min_vs_fullbar=MIN_VS_FULLBAR):
    """Вердикт K-11h-A по медианам одного батча. ХУДШАЯ голова, не средняя."""
    need = ("coarse24", "fullbar") + HICORA
    miss = [c for c in need if c not in med or med[c] is None]
    if miss:
        raise SystemExit(f"нет измерений для {miss}: вердикт K-11h-A "
                         f"невозможен")
    r_coarse = {s: med[s] / med["coarse24"] for s in HICORA}
    r_full = {s: med["fullbar"] / med[s] for s in HICORA}
    worst_c, worst_f = max(r_coarse.values()), min(r_full.values())
    ok_c, ok_f = worst_c <= max_vs_coarse, worst_f >= min_vs_fullbar
    return dict(ratio_vs_coarse=r_coarse, speedup_vs_fullbar=r_full,
                worst_vs_coarse=worst_c, worst_vs_fullbar=worst_f,
                ok_vs_coarse=bool(ok_c), ok_vs_fullbar=bool(ok_f),
                passed=bool(ok_c and ok_f),
                thresholds=dict(max_vs_coarse=max_vs_coarse,
                                min_vs_fullbar=min_vs_fullbar))


def paired_ratio_ci(a, b, n_boot=2000, seed=0):
    """ДИ отношения по ПАРНЫМ замерам: оба в одном повторе, дрейф общий.

    ПРОВЕРКА УСТОЙЧИВОСТИ, А НЕ ЗАМЕНА КРИТЕРИЯ. Зарегистрированная оценка —
    отношение медиан; здесь медиана отношений, и она в общем случае другая.
    Подменять одну другой задним числом нельзя, поэтому печатаются обе.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.size == 0:
        raise SystemExit("парные замеры разной длины: отношение не парное")
    r = a / b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, r.size, size=(n_boot, r.size))
    bs = np.median(r[idx], axis=1)
    return dict(median_of_ratios=float(np.median(r)),
                lo=float(np.percentile(bs, 2.5)),
                hi=float(np.percentile(bs, 97.5)), n_boot=int(n_boot))


def split_estimate(med, decode_med):
    """K-11h-B: оценка разрыва черновик/поправка. НЕ ГЕЙТ.

    ОЦЕНКА, А НЕ ЗАМЕР. Потоковой руки нет, прямо измерить «когда готов
    черновик» внутри hicora нельзя: 24 слоя уже исполнены к моменту взятия q0.
    Складывается из измеренных в предположении, что общий префикс из 12 слоёв
    считается один раз:

        t_draft       = T(joint12)                — черновик в исполнение
        t_refined_est = T(hicora) + T(decode)     — ВТОРОЕ декодирование
        delta         = t_refined_est - t_draft

    `delta < 50 мс` означает, что поправка успевает В ТЕЧЕНИЕ ОДНОГО ШАГА
    ПОСЛЕ ГОТОВНОСТИ ЧЕРНОВИКА (ActionCodec около 20 Гц). Это НЕ «36 мс от
    наблюдения»: сам черновик готов только через t_draft. Прежняя версия
    писала t_refined = T(hicora) без второго декодирования — для абсолютного
    времени потоковой политики это занижение.
    """
    if "joint12" not in med:
        return None
    out = {}
    for s in HICORA:
        if s not in med:
            continue
        t_ref = med[s] + decode_med
        out[s] = dict(t_draft=med["joint12"], t_refined_est=t_ref,
                      delta_ms=t_ref - med["joint12"],
                      extra_decode_ms=decode_med,
                      within_step=bool(t_ref - med["joint12"] < 50.0))
    return out


def registration(cfg, files=None, binding=None):
    """Чем прогон отличается от зарегистрированного протокола.

    Регистрация — это ТРИ условия сразу: режим и повторы как объявлено, все
    отпечатки входов вычислены, и веса те же, что проверял K-11e.
    """
    # FAIL-CLOSED ПО АРГУМЕНТАМ. Вызов без files или без результата binding
    # прежде возвращал registered=True: основной путь передаёт оба, но забыть
    # их мог бы любой будущий вызывающий.
    dev = []
    if binding is None:
        dev.append("привязка к протоколу K-11e не проверена")
    else:
        dev.extend(binding)
    if files is None:
        dev.append("отпечатки входов не переданы")
    else:
        for f, v in sorted(files.items()):
            if not v:
                dev.append(f"отпечаток {f} не вычислен")
    for k, want in REGISTERED.items():
        got = cfg.get(k)
        if isinstance(want, list):
            same = list(got or []) == want
        elif isinstance(want, float):
            same = got is not None and abs(float(got) - want) < 1e-12
        else:
            same = got == want
        if not same:
            dev.append(f"{k}: {got!r} вместо {want!r}")
    return dict(registered=not dev, deviations=dev)


def check_k11e_binding(k11e, files, ckpt):
    """Зарегистрированный K-11h обязан мерить ВЕСА, проверенные в K-11e.

    ЗАЧЕМ. `REGISTERED` фиксирует режим и число повторов, но не модель, не
    черновик и не головы: `registration` с чужим `ckpt` и пустыми sha
    возвращала registered=True. Протокол K-11h пишется прямо перед замером и
    защищает возобновление, но сам по себе предварительной регистрацией
    конфигурации не является. Ею является протокол K-11e — он записан ДО
    гейта и содержит ckpt, sha черновика и обеих голов.
    """
    bad = []
    if not isinstance(k11e, dict) or not k11e:
        return ["протокол K-11e не прочитан"]
    for f, key in (("joint12", "joint_sha1"), ("hicora_s0", "head_s0_sha1"),
                   ("hicora_s1", "head_s1_sha1")):
        got, want = files.get(f), k11e.get(key)
        if not got:
            bad.append(f"{f}: sha не вычислен (файла нет?)")
        elif not want:
            bad.append(f"{key} отсутствует в протоколе K-11e")
        elif got != want:
            bad.append(f"{f}: sha {got}, в K-11e {want}")
    # ОТСУТСТВИЕ ckpt В K-11e — ОТКАЗ, А НЕ СНЯТИЕ ПРОВЕРКИ. Прежде условие
    # было `if k11e.get("ckpt") and ...`, и протокол без этого поля вместе с
    # произвольной моделью давал зарегистрированный прогон.
    want_ckpt = k11e.get("ckpt")
    if not want_ckpt:
        bad.append("ckpt отсутствует в протоколе K-11e")
    elif ckpt != want_ckpt:
        bad.append(f"ckpt {ckpt!r}, в K-11e {want_ckpt!r}")
    for f in ("cache", "images", "script", "cfg_yaml"):
        if not files.get(f):
            bad.append(f"{f}: отпечаток не вычислен")
    return bad


def build_protocol(cfg, files, binding=None):
    p = dict(cfg)
    p["files"] = files
    p.update(registration(cfg, files, binding))
    return p


def verify_protocol(old, cur):
    """Расхождение с записанным протоколом — отказ, а не перезапись."""
    keys = set(old) | set(cur)
    diff = [k for k in sorted(keys)
            if k not in ("registered", "deviations")
            and str(old.get(k)) != str(cur.get(k))]
    if diff:
        raise SystemExit(
            f"ПРОТОКОЛ K-11h РАСХОДИТСЯ С ЗАПИСАННЫМ по полям {diff}.\n"
            f"  Сравнивать замеры разных условий нельзя. Для другой "
            f"конфигурации берите другой --proto и другой --out.")
    return True


def good_f_ph():
    """Полный набор непустых отпечатков — для самопроверки."""
    return dict(joint12="wj", hicora_s0="h0", hicora_s1="h1", cache="c",
                images="i", script="s", cfg_yaml="y", bar_py="b",
                joint12_vla="j", hicora_vla="h")


def check_orchestration(path=None):
    """Основной путь обязан вызывать тайминги и писать JSON.

    ЗАЧЕМ СТРУКТУРНАЯ ПРОВЕРКА. Вынос таймингов в отдельную функцию был
    сделан текстовым рефакторингом, и весь хвост `main` — вызов
    `timing_stage`, замер памяти, сбор и запись JSON — оказался ПОСЛЕ
    `return` внутри `timing_stage`, то есть недостижим. Команда при этом
    печатала протокол, порядок рук и завершалась с кодом 0, не измерив
    ничего и не создав `latency.json`: снаружи выглядело успехом. Ни
    компиляция, ни проверки чистых функций такого не видят, потому что
    ломается ОРКЕСТРАЦИЯ, а не вычисление.
    """
    import ast
    path = path or os.path.abspath(__file__)
    src = open(path).read()
    tree = ast.parse(src)
    fns = {n.name: n for n in tree.body
           if isinstance(n, ast.FunctionDef)}
    bad = []
    for need in ("main", "timing_stage", "mem_only", "equivalence_check"):
        if need not in fns:
            bad.append(f"нет функции {need}")
    if bad:
        raise SystemExit("СТРУКТУРА СЛОМАНА:\n    " + "\n    ".join(bad))
    ts = fns["timing_stage"]
    if not isinstance(ts.body[-1], ast.Return):
        bad.append("в timing_stage есть код ПОСЛЕ return — он недостижим")
    if sum(isinstance(x, ast.Return) for x in ts.body) != 1:
        bad.append("в теле timing_stage не ровно один return")
    m = ast.get_source_segment(src, fns["main"]) or ""
    for frag, why in (("timing_stage(", "main не вызывает timing_stage"),
                      ("json.dump(out", "main не пишет итоговый JSON"),
                      ("--mem-only", "main не запускает замер памяти"),
                      ("memory_complete",
                       "main не фиксирует полноту памяти")):
        if frag not in m:
            bad.append(why)
    # ВЕРДИКТ СЧИТАЕТСЯ В ТАЙМИНГАХ, А НЕ В main: требовать read_gate от main
    # было моей ошибкой, и проверка сама же её показала.
    t_src = ast.get_source_segment(src, ts) or ""
    for frag, why in (("read_gate(", "вердикт не считается"),
                      ("paired_ratio_ci(", "парное отношение не считается"),
                      ("split_estimate(", "K-11h-B не считается"),
                      ("rows_by_batch[bs] = row",
                       "строки батча не возвращаются наружу")):
        if frag not in t_src:
            bad.append(why)
    if "del S" in m:
        bad.append("в main остался del S: после выноса модели такой "
                   "переменной там нет, и это NameError")
    if bad:
        raise SystemExit("ОСНОВНОЙ ПУТЬ СЛОМАН:\n    " + "\n    ".join(bad)
                         + "\n  Прогон завершился бы с кодом 0, ничего не "
                           "измерив.")
    return True


def selftest():
    # ПЕРВОЙ — ПРОВЕРКА ОРКЕСТРАЦИИ: прочие проверки её поломку не видят.
    check_orchestration()
    base = dict(coarse24=83.6, fullbar=183.9, joint12=65.0,
                hicora_s0=90.5, hicora_s1=90.7)
    g = read_gate(base)
    assert g["passed"], g
    assert abs(g["worst_vs_coarse"] - 90.7 / 83.6) < 1e-9
    assert abs(g["worst_vs_fullbar"] - 183.9 / 90.7) < 1e-9
    assert not read_gate(dict(base, hicora_s1=95.0))["passed"]
    assert not read_gate(dict(base, fullbar=150.0))["ok_vs_fullbar"]
    assert read_gate(dict(base, hicora_s0=83.6 * 1.1,
                          hicora_s1=83.6 * 1.1))["passed"]
    for gone in ("coarse24", "fullbar", "hicora_s0", "hicora_s1"):
        try:
            read_gate({k: v for k, v in base.items() if k != gone})
        except SystemExit:
            pass
        else:
            raise AssertionError(f"отсутствие {gone} принято")
    try:
        read_gate(dict(base, hicora_s1=None))
    except SystemExit:
        pass
    else:
        raise AssertionError("None принят")
    # батч 10 из измеренного: против fullbar порог НЕ выполняется
    b10 = dict(coarse24=294.4, fullbar=563.4, joint12=259.1,
               hicora_s0=322.0, hicora_s1=322.0)
    g10 = read_gate(b10)
    assert g10["ok_vs_coarse"] and not g10["ok_vs_fullbar"], g10
    assert abs(g10["worst_vs_fullbar"] - 1.7497) < 1e-3

    # --- СБАЛАНСИРОВАННЫЙ ПОРЯДОК -------------------------------------------
    o = rotations(CONFIGS, 200)
    assert len(o) == 200 and check_balance(o, CONFIGS) == 40
    assert o[0] != o[1], "сдвига не произошло"
    try:
        rotations(CONFIGS, 7)
    except SystemExit:
        pass
    else:
        raise AssertionError("reps, не кратный числу рук, принят")
    try:
        check_balance([tuple(CONFIGS)] * 5, CONFIGS)
    except SystemExit:
        pass
    else:
        raise AssertionError("фиксированный порядок прошёл как баланс")

    # --- ПОЛНОТА ВЕСОВ В ОБЕ СТОРОНЫ ----------------------------------------
    class T:
        def __init__(self, shape):
            self.shape = shape

    ok = {"a": T((2, 2)), "b": T((3,))}
    assert check_joint_state(dict(ok), dict(ok))
    for bad, why in ((({"a": T((2, 2))}, dict(ok)), "в чекпойнте нет ключа"),
                     ((dict(ok), {"a": T((2, 2))}), "лишний ключ"),
                     (({"a": T((2, 3)), "b": T((3,))}, dict(ok)), "форма")):
        try:
            check_joint_state(bad[0], bad[1])
        except SystemExit:
            pass
        else:
            raise AssertionError(f"принято: {why}")

    # --- K-11h-B: второе декодирование учтено -------------------------------
    sp = split_estimate(base, 10.57)
    s0 = sp["hicora_s0"]
    assert abs(s0["t_refined_est"] - (90.5 + 10.57)) < 1e-9
    assert abs(s0["delta_ms"] - (90.5 + 10.57 - 65.0)) < 1e-9
    assert s0["within_step"] and s0["delta_ms"] < 50.0
    assert abs(s0["t_draft"] - 65.0) < 1e-9
    assert not split_estimate(b10, 12.22)["hicora_s0"]["within_step"]
    assert split_estimate({"hicora_s0": 1.0}, 2.0) is None

    # --- ПАРНОЕ ОТНОШЕНИЕ ---------------------------------------------------
    rng = np.random.default_rng(0)
    drift = np.linspace(1.0, 1.3, 400)          # общий дрейф частоты
    a = 90.0 * drift + rng.normal(0, 0.3, 400)
    b = 83.0 * drift + rng.normal(0, 0.3, 400)
    ci = paired_ratio_ci(a, b)
    assert abs(ci["median_of_ratios"] - 90.0 / 83.0) < 0.01, ci
    assert ci["lo"] < 90.0 / 83.0 < ci["hi"]
    assert np.std(a / b) * 10 < np.std(a), "парность не сняла общий дрейф"
    try:
        paired_ratio_ci([1.0, 2.0], [1.0])
    except SystemExit:
        pass
    else:
        raise AssertionError("непарные длины приняты")

    # --- РЕГИСТРАЦИЯ --------------------------------------------------------
    good = dict(REGISTERED)
    # БЕЗ files И binding — НЕ зарегистрирован: fail-closed по аргументам.
    r0 = registration(good)
    assert not r0["registered"] and len(r0["deviations"]) == 2, r0
    assert not registration(good, good_f_ph())["registered"]
    assert not registration(good, None, [])["registered"]
    for k, v in (("reps", 1), ("warmup", 0), ("batches", [1]),
                 ("dtype", "bfloat16"), ("pos_offset", 0),
                 ("weight_mode", "uniform-trunk"), ("max_vs_coarse", 1.5)):
        r = registration(dict(good, **{k: v}))
        assert not r["registered"] and any(k in d for d in r["deviations"]), k
    # --- ПРИВЯЗКА К ВЕСАМ K-11e ---------------------------------------------
    # ДЫРА, КОТОРУЮ ЭТО ЗАКРЫВАЕТ: registration с ЧУЖИМ ckpt и пустыми sha
    # возвращала registered=True. Режим и число повторов ничего не говорят о
    # том, ЧТО именно мерилось.
    K11E = dict(ckpt="A/B", joint_sha1="wj", head_s0_sha1="h0",
                head_s1_sha1="h1")
    good_f = good_f_ph()
    assert check_k11e_binding(K11E, good_f, "A/B") == []
    assert registration(good, good_f,
                        check_k11e_binding(K11E, good_f, "A/B"))["registered"]
    for mut, why in ((dict(joint12="ИНОЙ"), "чужой черновик"),
                     (dict(hicora_s0="ИНАЯ"), "чужая голова s0"),
                     (dict(hicora_s1="ИНАЯ"), "чужая голова s1"),
                     (dict(joint12=None), "sha черновика не вычислен"),
                     (dict(images=None), "нет отпечатка картинок"),
                     (dict(cfg_yaml=None), "нет отпечатка конфига"),
                     (dict(cache=None), "нет отпечатка кэша")):
        b = check_k11e_binding(K11E, dict(good_f, **mut), "A/B")
        assert b, f"принято: {why}"
        assert not registration(good, dict(good_f, **mut), b)["registered"]
    assert check_k11e_binding(K11E, good_f, "ДРУГАЯ/МОДЕЛЬ"), "чужой ckpt"
    assert not registration(good, good_f, check_k11e_binding(
        K11E, good_f, "ДРУГАЯ/МОДЕЛЬ"))["registered"]
    assert check_k11e_binding({}, good_f, "A/B"), "пустой протокол K-11e"
    for gone in ("joint_sha1", "head_s0_sha1", "head_s1_sha1", "ckpt"):
        assert check_k11e_binding({k: v for k, v in K11E.items() if k != gone},
                                  good_f, "A/B"), f"нет {gone} в K-11e"

    p = build_protocol(good, good_f, [])
    assert verify_protocol(p, build_protocol(good, good_f, []))
    try:
        verify_protocol(p, build_protocol(dict(good, reps=100), good_f, []))
    except SystemExit:
        pass
    else:
        raise AssertionError("смена reps при возобновлении принята")
    # ОКРУЖЕНИЕ ТОЖЕ ЗАПЕРТО: другая карта или другой torch — отказ.
    e0 = dict(good, env=dict(gpu="V100", torch="2.4.1", cuda="12.4"))
    for mut in (dict(gpu="A100"), dict(torch="2.5.0"), dict(cuda="12.1")):
        try:
            verify_protocol(build_protocol(e0, good_f, []),
                            build_protocol(dict(e0, env=dict(e0["env"], **mut)),
                                           good_f, []))
        except SystemExit:
            pass
        else:
            raise AssertionError(f"смена окружения принята: {mut}")

    # --- опора K-9i: порог 0.60 ей противоречил -----------------------------
    assert abs(65.1 / 84.7 - 0.7686) < 1e-3
    assert abs(261.0 / 296.8 - 0.8794) < 1e-3

    print("самопроверка k11h пройдена: основной путь вызывает тайминги и "
          "пишет JSON,\n  вердикт по ХУДШЕЙ голове, батч 10 "
          "обязателен\n  в зарегистрированном протоколе, порядок рук "
          "сбалансирован точно, веса\n  сверяются в обе стороны, K-11h-B "
          "учитывает второе декодирование,\n  парное отношение снимает общий "
          "дрейф, регистрация привязана к весам K-11e\n  и отпечаткам входов, "
          "смена условий или окружения при возобновлении — отказ")


# ============================ ИЗМЕРЕНИЕ ====================================

def build_stack(args, arms, dev, dt):
    """Собрать ровно то, что нужно перечисленным рукам.

    ПОЧЕМУ ПАРАМЕТРИЗОВАНО. Для замера памяти отдельной руки нельзя загружать
    чужие веса: fullbar и coarse24 не вызывают init_joint_fast вовсе, и их пик
    памяти — это пик БЕЗ головы черновика.
    """
    import torch

    import actioncodec  # noqa: F401
    import joint12_vla as jv
    from joint12_vla import make_joint12_class
    from smolvla.bar import SmolVLABlockwiseAR
    from utils import (STATE_Q01, STATE_Q99, VisionLanguageActionProcessor,
                       dict_apply, get_cfg, prompt_template, seed_everything)
    import hicora_vla as hv
    import k9h_multiarm_gate as k9h
    import k11a_build_hicora_cache as k11a
    import k11e_protocol as kp

    seed_everything(0)
    cfg = get_cfg(os.path.join(os.path.abspath(args.root), args.cfg_path))
    cfg.TRAINING.ckpt_dir = args.ckpt
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = args.ckpt

    z = np.load(args.cache, allow_pickle=True)
    tsk = z["task"]
    IMG = np.load(args.cache + ".images.npy", mmap_mode="r")
    batches = [int(x) for x in str(args.batches).split(",")]
    st = np.zeros((max(batches), len(STATE_Q01)), np.float64)
    st_n = (st - STATE_Q01) / (STATE_Q99 - STATE_Q01) * 2.0 - 1.0

    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")

    def build(b):
        image = torch.from_numpy(np.asarray(IMG[:b]))
        msgs = []
        for i in range(b):
            m = prompt_template(
                st_n[i], None, str(tsk[i]),
                mode=cfg.MODEL.vla_processor.kwargs.mode,
                action_vocab_size=cfg.MODEL.action_processor.vocab_size,
                action_token_len=cfg.MODEL.action_processor.token_len)
            m[1]["content"] = m[1]["content"][1:]
            msgs.append(m)
        texts = proc.apply_chat_template(msgs, add_generation_prompt=True)
        bt = proc(text=texts, images=[[image[k].numpy()] for k in range(b)],
                  return_tensors="pt", padding=True, padding_side="left",
                  action_processor_kwargs={"embodiment_ids": 0})
        return dict_apply(lambda x: x.to(dev, dt), bt)

    need_joint = any(a in NEEDS_JOINT for a in arms)
    need_hic = any(a in HICORA for a in arms)

    Cls = make_joint12_class(SmolVLABlockwiseAR)
    model = Cls.from_pretrained(**cfg.MODEL.vlm.kwargs).to(dev, dt).eval()
    import copy
    res_norm_orig = (copy.deepcopy(model.action_expert.norm)
                     if need_hic else None)
    base_state, j_state, own, joint_sha = None, None, None, None
    if need_joint:
        model.init_joint_fast(depth=12, head_dtype=dt)
        own = dict(model.named_parameters())
        base_state = {k: own[k].detach().clone()
                      for k in own if own[k].requires_grad}
        j_obj = torch.load(args.joint12, map_location="cpu",
                           weights_only=False)
        joint_sha = k9h.file_sha12(args.joint12)
        # DTYPE ВЕСОВ ЧЕРНОВИКА — ГЛАВНЫЙ ПЕРЕКЛЮЧАТЕЛЬ ЭТОГО СКРИПТА.
        w_dt = torch.float32 if args.weight_mode == "as-executed" else dt
        j_state = {k: v.to(dev, w_dt) for k, v in j_obj["state"].items()}
        # В режиме uniform держим И fp32-состояние: без него нечем показать,
        # что смена dtype не изменила саму политику.
        j_state_ref = (None if args.weight_mode == "as-executed"
                       else {k: v.to(dev, torch.float32)
                             for k, v in j_obj["state"].items()})
        check_joint_state(j_state, base_state)
        for k, v in j_state.items():
            if not torch.isfinite(v).all():
                raise SystemExit(f"в весах Joint12 нечисловое значение: {k}")
        print(f"  черновик Joint12: {len(j_state)} тензоров, sha {joint_sha}, "
              f"dtype весов {w_dt}")

    ac = proc.action_processor
    codec = (ac if hasattr(ac, "vq") else getattr(ac, "codec", None))
    if codec is None or not hasattr(codec, "vq"):
        raise SystemExit("не нашёл квантователь в action_processor")
    codec = codec.to(dev).eval()
    with torch.no_grad():
        ii = torch.arange(int(codec.vocab_size), device=dev).unsqueeze(0)
        E = torch.stack([q.out_project(q.decode_code(ii))[0]
                         for q in codec.vq.quantizers]).float().to(dev)

    heads, head_sha, rn_sha, head_cfg = {}, {}, None, {}
    if need_hic:
        model.__class__ = hv.make_hicora_class(type(model))
        model.set_codebooks(E)
        model.set_res_norm(res_norm_orig.to(dev))
        model.taps, model.q0_depth = (12, 18, 24), 12
        model.n_layers_total = len(model.action_expert.layers)
        if max(model.taps) != model.n_layers_total:
            raise SystemExit(f"последний отвод {max(model.taps)} против "
                             f"{model.n_layers_total} слоёв")
        h = hashlib.sha1()
        for k_ in sorted(model.res_norm.state_dict()):
            v_ = model.res_norm.state_dict()[k_]
            h.update(k_.encode())
            h.update(np.ascontiguousarray(
                v_.detach().float().cpu().numpy()).tobytes())
        rn_sha = h.hexdigest()[:12]
        for nm, path in (("hicora_s0", args.hicora_s0),
                         ("hicora_s1", args.hicora_s1)):
            if nm not in arms:
                continue
            o = torch.load(path, map_location="cpu", weights_only=False)
            k9h.check_hicora_ckpt(o, nm, args.expect_hicora_target)
            pref = o["cache"]
            bp, rp = pref + ".basis.npy", pref + ".rho.npy"
            mp = pref + ".meta.json"
            for f_ in (bp, rp, mp):
                if not os.path.exists(f_):
                    raise SystemExit(f"нет {f_}")
            for f_, want_, lbl in ((bp, o["basis_sha1"], "базис"),
                                   (rp, o["rho_sha1"], "предел")):
                got_ = k9h.file_sha12(f_)
                if got_ != want_:
                    raise SystemExit(f"{lbl} sha {got_}, голова обучена на "
                                     f"{want_}")
            if rn_sha != o["res_norm_sha1"]:
                raise SystemExit(f"res_norm sha {rn_sha}, голова {nm} обучена "
                                 f"на {o['res_norm_sha1']}")
            meta = json.load(open(mp))
            k9h.check_hicora_meta(meta, args.ckpt, joint_sha,
                                  k9h.file_sha12(hv.__file__),
                                  k9h.file_sha12(jv.__file__))
            k11a.check_fingerprints(meta, dict(
                codebooks_sha1=hashlib.sha1(np.ascontiguousarray(
                    E.cpu().numpy().astype(np.float32)
                ).tobytes()).hexdigest()[:12],
                decoder_probe=k11a.decoder_probe(codec, E, dev),
                codec_state_sha1=k11a.state_sha1(codec)))
            B = np.load(bp).astype(np.float32)
            rho = np.load(rp).astype(np.float32)
            hd = hv.make_residual_head()(
                int(model.fast_head.in_features), int(E.shape[-1]),
                rank=int(o["rank"]), hidden=int(o.get("hidden", 512)),
                proj=int(o.get("proj", 64))).to(dev)
            hd.set_basis(torch.as_tensor(B))
            hd.set_rho(torch.as_tensor(rho))
            stray = [k for k in o["state"]
                     if not k.startswith("hicora_head.")]
            if stray:
                raise SystemExit(f"в чекпойнте {nm} ключи вне hicora_head.: "
                                 f"{stray[:5]}")
            st_h = {k[len("hicora_head."):]: v for k, v in o["state"].items()}
            want_h = {k for k in hd.state_dict()
                      if k.startswith(("proj.", "net."))}
            if set(st_h) != want_h:
                raise SystemExit(f"набор весов головы {nm} не совпал")
            with torch.no_grad():
                for k, v in st_h.items():
                    hd.state_dict()[k].copy_(v.to(dev, torch.float32))
            hd.eval()
            heads[nm] = hd
            head_sha[nm] = k9h.file_sha12(path)
            head_cfg[nm] = kp.head_config(o)
            print(f"  голова {nm}: sha {head_sha[nm]}, ранг {o['rank']}, "
                  f"сид {o['seed']}, эпоха {o.get('selected_epoch')}")
        if len(head_sha) == 2:
            if head_sha["hicora_s0"] == head_sha["hicora_s1"]:
                raise SystemExit("обе головы — один файл: это не две руки")
            # РЕПЛИКАЦИЯ СВЕРЯЕТСЯ ТЕМ ЖЕ КОДОМ, ЧТО В K-11e. Прежде головы
            # проверялись по отдельности, и s0/s1 с разными cache/rank/lr/wd
            # прошли бы как «две руки одного эксперимента».
            kp.check_replication(head_cfg["hicora_s0"], head_cfg["hicora_s1"])
            print("  репликация s0/s1: различаются только сидом")
    return dict(torch=torch, model=model, own=own, base_state=base_state,
                j_state=j_state, j_state_ref=locals().get("j_state_ref"),
                codec=codec, E=E, heads=heads,
                build=build, batches=batches, joint_sha=joint_sha,
                head_sha=head_sha, rn_sha=rn_sha)


def make_runner(S, args):
    torch = S["torch"]
    model, own, E, codec = S["model"], S["own"], S["E"], S["codec"]

    @contextlib.contextmanager
    def only_blocks(n):
        saved = model.num_blocks
        try:
            model.num_blocks = n
            yield
        finally:
            model.num_blocks = saved

    def apply_weights(name):
        """Веса руки. Копирование дешёвое и ВНЕ измеряемого участка."""
        if own is None:
            return
        src = S["base_state"] if name in BASE_W else S["j_state"]
        if src is None:
            return
        with torch.no_grad():
            for k, v in src.items():
                own[k].data = v
        if name in HICORA:
            model.hicora_head = S["heads"][name]

    def decode_codes(codes, n_lv):
        K = codes.reshape(-1, n_lv, N_POS)
        zq = E[0][torch.as_tensor(K[:, 0, :]).long().to(E.device)]
        for j in range(1, n_lv):
            zq = zq + E[j][torch.as_tensor(K[:, j, :]).long().to(E.device)]
        x, _ = codec._decode(zq, embodiment_ids=0)
        return x[..., :7]

    def decode_latent(zl):
        x, _ = codec._decode(zl.float(), embodiment_ids=0)
        return x[..., :7]

    ac16 = torch.autocast("cuda", dtype=torch.float16)

    def run(name, batch):
        if name == "fullbar":
            with torch.no_grad():
                t = model.generate(**batch, position_offset=args.pos_offset,
                                   do_sample=False)
            return ("codes", t.cpu().numpy(), N_LEVEL)
        if name == "coarse24":
            with torch.no_grad(), only_blocks(1):
                t = model.generate(**batch, position_offset=args.pos_offset,
                                   do_sample=False)
            return ("codes", t[:, :N_POS].cpu().numpy(), 1)
        with torch.no_grad(), ac16:
            v, p = model.build_inputs(position_offset=args.pos_offset, **batch)
            if name == "joint12":
                o = model.forward_joint_fast(
                    vlm_inputs_embeds=v,
                    attention_mask=batch.get("attention_mask"), position_ids=p)
                return ("codes", o["pred_codes"].cpu().numpy(), 1)
            o = model.forward_hicora(
                vlm_inputs_embeds=v,
                attention_mask=batch.get("attention_mask"), position_ids=p)
            if int(o["layers_run"]) != 24:
                raise SystemExit(f"{name}: исполнено {o['layers_run']} слоёв "
                                 f"вместо 24 — мерится не та политика")
            return ("latent", o["z"], 1)

    def do_decode(kind, payload, n_lv):
        return (decode_codes(payload, n_lv) if kind == "codes"
                else decode_latent(payload))

    return apply_weights, run, do_decode


def equivalence_check(S, args, batch, arms=HICORA,
                      code_frac=0.01, act_rel=1e-2):
    """Смена dtype не должна менять политику: q0, dz и действия те же.

    ЗАЧЕМ. Режим `uniform-trunk` отвечает на архитектурный вопрос, но не
    то, что исполнял гейт K-11e. Прежде чем принимать его числа, надо
    показать, что он считает то же самое. Без этой проверки «архитектурная
    стоимость» могла бы относиться к другой политике.

    Допуски: коды q0 дискретны и на границе могут перевернуться, поэтому
    разрешена доля `code_frac`; действия сравниваются относительной нормой.
    """
    torch = S["torch"]
    model, own = S["model"], S["own"]
    if S.get("j_state_ref") is None:
        return None

    def once(state, arm):
        with torch.no_grad():
            for k, v in state.items():
                own[k].data = v
        model.hicora_head = S["heads"][arm]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            v_, p_ = model.build_inputs(position_offset=args.pos_offset,
                                        **batch)
            o = model.forward_hicora(
                vlm_inputs_embeds=v_,
                attention_mask=batch.get("attention_mask"), position_ids=p_)
        x, _ = S["codec"]._decode(o["z"].float(), embodiment_ids=0)
        return (o["q0"].detach().cpu().numpy(),
                o["dz"].detach().float().cpu().numpy(),
                x[..., :7].detach().float().cpu().numpy())

    res, bad = {}, []
    for arm in arms:
        if arm not in S["heads"]:
            continue
        q_r, dz_r, a_r = once(S["j_state_ref"], arm)
        q_u, dz_u, a_u = once(S["j_state"], arm)
        frac = float((q_r != q_u).mean())
        rel_a = float(np.linalg.norm(a_u - a_r)
                      / (np.linalg.norm(a_r) or 1.0))
        rel_dz = float(np.linalg.norm(dz_u - dz_r)
                       / (np.linalg.norm(dz_r) or 1.0))
        ok = frac <= code_frac and rel_a <= act_rel
        res[arm] = dict(q0_diff_frac=frac, dz_rel=rel_dz, act_rel=rel_a,
                        n_inputs=int(np.asarray(q_r).shape[0]), ok=bool(ok))
        print(f"  совпадение на фиксированных входах "
              f"({res[arm]['n_inputs']} шт), {arm}: коды q0 разошлись у "
              f"{frac:.3%} (порог {code_frac:.1%}), dz отн. {rel_dz:.2e}, "
              f"действия отн. {rel_a:.2e} (порог {act_rel:.0e}) -> "
              f"{'да' if ok else 'НЕТ'}")
        if not ok:
            bad.append(arm)
    if bad:
        raise SystemExit(
            f"СМЕНА DTYPE СТВОЛА ИЗМЕНИЛА ПОЛИТИКУ на {bad}. Числа режима "
            f"uniform-trunk\n  относились бы к другой политике.")
    # FP32-КОПИЯ УДАЛЯЕТСЯ ДО ЗАМЕРА: она нужна была только для сверки.
    import gc
    S["j_state_ref"] = None
    gc.collect()
    torch.cuda.empty_cache()
    print("  ЭТО СОВПАДЕНИЕ НА ФИКСИРОВАННЫХ ВХОДАХ, а не эквивалентность "
          "политики\n  на распределении LIBERO: проверены только эти входы.")
    return dict(per_arm=res, code_frac=code_frac, act_rel=act_rel)


def mem_only(args):
    """Пик памяти ОДНОЙ руки в отдельном процессе."""
    import torch
    arm = args.mem_only
    dev = torch.device(args.device)
    dt = getattr(torch, args.dtype)
    S = build_stack(args, (arm,), dev, dt)
    apply_weights, run, do_decode = make_runner(S, args)
    batch = S["build"](int(str(args.batches).split(",")[0]))
    apply_weights(arm)
    # ВСПОМОГАТЕЛЬНЫЕ КОПИИ УДАЛЯЮТСЯ ДО ЗАМЕРА. Иначе процесс держит
    # base_state (лишние ~1.7 ГиБ в fp16) и, в uniform-trunk, ещё и fp32
    # j_state_ref (~3.5 ГиБ) — у реально развёрнутой политики их нет, и
    # peak_mib не был бы памятью развёртывания. Веса уже переданы параметрам
    # по ссылке, поэтому удаление словарей их не освобождает.
    import gc
    S["base_state"] = None
    S["j_state_ref"] = None
    S["j_state"] = None
    gc.collect()
    torch.cuda.empty_cache()
    for _ in range(3):
        k, pl, nl = run(arm, batch)
        do_decode(k, pl, nl)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)
    for _ in range(5):
        k, pl, nl = run(arm, batch)
        do_decode(k, pl, nl)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(dev) / 2 ** 20
    print("MEMJSON " + json.dumps({"arm": arm, "peak_mib": peak}))


def timing_stage(args, dev, dt, orders, batches):
    """Все тайминги. ОТДЕЛЬНОЙ ФУНКЦИЕЙ РАДИ ОСВОБОЖДЕНИЯ КАРТЫ.

    `del S` в main не освобождал модель: её держали замыкания `apply_weights`
    (замыкает S целиком), `run` (model) и `do_decode` (codec, E), а также
    локальные `batch` и последний результат прохода. Дочерние процессы замера
    памяти поднимались рядом с живой родительской моделью. По возврату из
    функции все эти ссылки исчезают сами.

    Возвращает (rows_by_batch, batches_out, equivalence, meta).
    """
    import torch
    S = build_stack(args, CONFIGS, dev, dt)
    apply_weights, run, do_decode = make_runner(S, args)

    print(f"\nруки: {', '.join(CONFIGS)}")
    print("  ВНИМАНИЕ: потоковая политика здесь НЕ мерится — её нет в коде, "
          "и успех\n  у неё был бы свой.")
    if args.weight_mode == "as-executed":
        print("  DTYPE as-executed: веса черновика в fp32 под autocast, как в "
              "K-11e.\n  Пик памяти рук joint12/hicora включает fp32-копию и "
              "АРХИТЕКТУРНЫМ НЕ ЯВЛЯЕТСЯ.")

    meta = dict(joint_sha1=S["joint_sha"],
                head_sha1=S["head_sha"], res_norm_sha1=S["rn_sha"])
    batches_out = {}

    if args.weight_mode == "uniform-trunk":
        # НА САМОМ БОЛЬШОМ БАТЧЕ: это все фиксированные входы сразу, а не
        # первый. Одного входа хватило бы лишь на грубую проверку связности.
        out_eq = equivalence_check(S, args, S["build"](max(batches)))
    else:
        out_eq = None

    rows_by_batch = {}
    for bs in batches:
        batch = S["build"](bs)
        print(f"\n=== батч {bs} ===")
        for name in CONFIGS:
            apply_weights(name)
            for _ in range(args.warmup):
                k, pl, nl = run(name, batch)
                do_decode(k, pl, nl)
            torch.cuda.synchronize()
        tm = {c: [] for c in CONFIGS}
        td = {c: [] for c in CONFIGS}
        for r, order in enumerate(orders):
            for name in order:
                apply_weights(name)          # ВНЕ замера
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                k, pl, nl = run(name, batch)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                do_decode(k, pl, nl)
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                tm[name].append((t1 - t0) * 1000)
                td[name].append((t2 - t1) * 1000)
            if (r + 1) % 50 == 0:
                print(f"    повтор {r + 1}/{args.reps}", flush=True)

        row = {}
        for name in CONFIGS:
            tot = [m + d for m, d in zip(tm[name], td[name])]
            row[name] = dict(model_ms=stats(tm[name]),
                             decode_ms=stats(td[name]),
                             total_ms=stats(tot), peak_mib=None,
                             raw_model_ms=tm[name], raw_decode_ms=td[name])
        med = {c: row[c]["total_ms"]["median"] for c in CONFIGS}
        gate = read_gate(med, MAX_VS_COARSE, MIN_VS_FULLBAR)
        tot_of = {c: [m + d for m, d in zip(tm[c], td[c])] for c in CONFIGS}
        pair = {s: dict(
            vs_coarse=paired_ratio_ci(tot_of[s], tot_of["coarse24"],
                                      args.n_boot),
            vs_fullbar=paired_ratio_ci(tot_of["fullbar"], tot_of[s],
                                       args.n_boot)) for s in HICORA}
        sp = split_estimate(med, float(np.median(
            [row[s]["decode_ms"]["median"] for s in HICORA])))

        print(f"\n  {'рука':<12}{'слоёв':>7}{'прох':>6}{'декод':>7}"
              f"{'медиана':>10}{'сред':>9}{'p95':>9}")
        for name in CONFIGS:
            r_ = row[name]
            print(f"  {name:<12}{LAYERS[name]:>7}{PASSES[name]:>6}"
                  f"{DECODES[name]:>7}{r_['total_ms']['median']:>10.1f}"
                  f"{r_['total_ms']['mean']:>9.1f}"
                  f"{r_['total_ms']['p95']:>9.1f}")
        print("\n  против coarse24 (отношение медиан): "
              + ", ".join(f"{s} {v:.3f}x" for s, v
                          in sorted(gate["ratio_vs_coarse"].items()))
              + f"; худшая {gate['worst_vs_coarse']:.3f}, порог "
                f"<= {MAX_VS_COARSE} -> "
                f"{'да' if gate['ok_vs_coarse'] else 'НЕТ'}")
        print("  против fullbar  (отношение медиан): "
              + ", ".join(f"{s} {v:.2f}x" for s, v
                          in sorted(gate["speedup_vs_fullbar"].items()))
              + f"; худшая {gate['worst_vs_fullbar']:.2f}, порог "
                f">= {MIN_VS_FULLBAR} -> "
                f"{'да' if gate['ok_vs_fullbar'] else 'НЕТ'}")
        print("  ПАРНОЕ отношение (проверка устойчивости, критерий НЕ "
              "подменяет):")
        for s in sorted(pair):
            a_, b_ = pair[s]["vs_coarse"], pair[s]["vs_fullbar"]
            print(f"    {s}: /coarse24 {a_['median_of_ratios']:.3f} "
                  f"[{a_['lo']:.3f}, {a_['hi']:.3f}]   "
                  f"fullbar/ {b_['median_of_ratios']:.2f} "
                  f"[{b_['lo']:.2f}, {b_['hi']:.2f}]")
        if sp:
            for s in sorted(sp):
                d = sp[s]
                print(f"  K-11h-B, {s}: черновик готов через "
                      f"{d['t_draft']:.1f} мс, исправленный — через "
                      f"{d['t_refined_est']:.1f} мс,\n    то есть поправка "
                      f"приходит спустя {d['delta_ms']:.1f} мс ПОСЛЕ "
                      f"черновика"
                      + ("  — внутрь шага 50 мс" if d["within_step"]
                         else "  — НЕ внутрь шага 50 мс"))
            print("  ЭТО ОЦЕНКА, НЕ ЗАМЕР: потоковой руки не существует.")

        rows_by_batch[bs] = row

        if bs == PRIMARY_BATCH:
            verd = "ПРОЙДЕН" if gate["passed"] else "НЕ ПРОЙДЕН"
            print(f"\n  K-11h-A ({verd}) — первичный критерий, батч {bs}")
        else:
            print(f"\n  батч {bs} — сопутствующий результат, в критерий НЕ "
                  f"входит"
                  + ("" if gate["passed"]
                     else "; на нём порог НЕ выполняется"))
        batches_out[str(bs)] = dict(
            rows=row, median_ms=med, gate=gate, paired=pair,
            split_estimate=sp, is_primary=bool(bs == PRIMARY_BATCH))

    return rows_by_batch, batches_out, out_eq, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--cache", default="data/k9_teacher_150k.npz")
    ap.add_argument("--joint12", default="data/k9d_ep3.pt")
    ap.add_argument("--hicora-s0",
                    default="data/k11d/d1_mlp_coef_0.001_wd0_s0.pt")
    ap.add_argument("--hicora-s1",
                    default="data/k11d/d1_mlp_coef_0.001_wd0_s1.pt")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--batches", default="1,10")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--pos-offset", type=int, default=4)
    ap.add_argument("--weight-mode", choices=["as-executed", "uniform-trunk"],
                    default="as-executed",
                    help="as-executed: веса черновика в fp32 под autocast, "
                         "РОВНО как исполнял гейт K-11e (зарегистрировано). "
                         "uniform-trunk: единый dtype СТВОЛА (голова, книги "
                         "и декодер остаются fp32) — отвечает про АРХИТЕКТУРУ, "
                         "но это другая реализация, и прогон помечается НЕ "
                         "зарегистрированным")
    ap.add_argument("--expect-hicora-target", default="coef")
    ap.add_argument("--proto", default="data/k11h/protocol.json")
    ap.add_argument("--k11e-proto", default="data/k11e/protocol.json",
                    help="протокол K-11e. Зарегистрированный K-11h обязан "
                         "мерить ТЕ ЖЕ веса, что проверял гейт; без совпадения "
                         "sha черновика, обеих голов и ckpt прогон "
                         "помечается НЕ зарегистрированным")
    ap.add_argument("--mem-only", default=None,
                    help="внутренний режим: пик памяти одной руки")
    ap.add_argument("--allow-unregistered", action="store_true",
                    help="начать прогон, не совпадающий с зарегистрированным "
                         "протоколом. Нужен для uniform-trunk и любых иных "
                         "условий; без него такой прогон отказывается "
                         "стартовать, а не тратит часы впустую")
    ap.add_argument("--no-mem", action="store_true",
                    help="не мерить память вовсе")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--allow-busy-gpu", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.ckpt:
        raise SystemExit("нужен --ckpt")

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        raise SystemExit(f"нет каталога ActionCodec: {root}")
    sys.path.insert(0, root)

    if args.mem_only:
        mem_only(args)
        return

    import torch
    import k9h_multiarm_gate as k9h

    dev = torch.device(args.device)
    dt = getattr(torch, args.dtype)
    if dev.type != "cuda":
        raise SystemExit("латентность меряется на GPU")
    free_b, _ = torch.cuda.mem_get_info(dev)
    if free_b / 2 ** 30 < 20 and not args.allow_busy_gpu:
        raise SystemExit(f"на {dev} свободно {free_b / 2 ** 30:.1f} ГБ: чужая "
                         f"нагрузка исказит замер. --allow-busy-gpu осознанно")

    batches = [int(x) for x in args.batches.split(",")]
    cfg = dict(batches=batches, reps=args.reps, warmup=args.warmup,
               dtype=args.dtype, pos_offset=args.pos_offset,
               weight_mode=args.weight_mode,
               max_vs_coarse=MAX_VS_COARSE, min_vs_fullbar=MIN_VS_FULLBAR,
               ckpt=args.ckpt, cache=args.cache)
    # ОТПЕЧАТОК КАРТИНОК ОТДЕЛЬНО: хешировался только .npz, а изображения
    # берутся из .images.npy — незахешированного файла рядом. Подмена картинок
    # меняла бы формы входа и время, оставаясь незамеченной.
    import smolvla.bar as _bar
    import joint12_vla as _jv
    import hicora_vla as _hv
    files = {k: (k9h.file_sha12(v) if v and os.path.exists(v) else None)
             for k, v in (("joint12", args.joint12),
                          ("hicora_s0", args.hicora_s0),
                          ("hicora_s1", args.hicora_s1),
                          ("cache", args.cache),
                          ("images", args.cache + ".images.npy"),
                          ("bar_py", _bar.__file__),
                          ("joint12_vla", _jv.__file__),
                          ("hicora_vla", _hv.__file__),
                          ("cfg_yaml", os.path.join(root, args.cfg_path)),
                          ("script", os.path.abspath(__file__)))}
    # ОКРУЖЕНИЕ ВХОДИТ В ПРОТОКОЛ: латентность от него зависит напрямую, и
    # возобновление на другой карте или другой версии torch — отказ.
    cfg["env"] = dict(gpu=torch.cuda.get_device_name(dev),
                      torch=torch.__version__, cuda=str(torch.version.cuda),
                      cfg_path=args.cfg_path)
    k11e_path = args.k11e_proto
    k11e = {}
    if os.path.exists(k11e_path):
        k11e = json.load(open(k11e_path))
    binding = check_k11e_binding(k11e, files, args.ckpt)
    if binding and args.weight_mode == "as-executed":
        print(f"  ПРИВЯЗКА К K-11e НЕ СОШЛАСЬ ({k11e_path}):")
        for b in binding:
            print(f"    {b}")
    proto = build_protocol(cfg, files, binding)
    reg = registration(cfg, files, binding)
    if os.path.exists(args.proto):
        verify_protocol(json.load(open(args.proto)), proto)
        print(f"  протокол сверен: {args.proto}")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.proto)) or ".",
                    exist_ok=True)
        tmp = args.proto + ".tmp"
        json.dump(proto, open(tmp, "w"), ensure_ascii=False, indent=1)
        os.replace(tmp, args.proto)
        print(f"  протокол записан: {args.proto}")
    if reg["registered"]:
        print("  прогон ЗАРЕГИСТРИРОВАННЫЙ")
    else:
        print("  прогон НЕ ЗАРЕГИСТРИРОВАННЫЙ, отклонения:")
        for d in reg["deviations"]:
            print(f"    {d}")
        # ОТКАЗ ДО ПОСТРОЙКИ МОДЕЛИ. Прежде отклонения печатались, и расчёт
        # продолжался несколько часов. Под nohup «увидеть первые строки и не
        # запускать дальше» невозможно — он уже идёт.
        if not args.allow_unregistered:
            raise SystemExit(
                "  ОТКАЗ: незарегистрированный прогон не начинается сам. "
                "Для режима\n  uniform-trunk или иных условий передайте "
                "--allow-unregistered осознанно.")
        print("  продолжаю по --allow-unregistered")

    orders = rotations(CONFIGS, args.reps)
    per_pos = check_balance(orders, CONFIGS)
    print(f"  порядок рук: циклический сдвиг, каждая рука в каждой позиции "
          f"{per_pos} раз")

    rows_by_batch, out_batches, out_eq, meta = timing_stage(
        args, dev, dt, orders, batches)
    # ПОСЛЕ ВОЗВРАТА ссылок на модель, кодек и батчи не осталось: они были
    # локальными в timing_stage. Проверяем это числом, а не на веру.
    import gc as _gc
    _gc.collect()
    torch.cuda.empty_cache()
    left = torch.cuda.memory_allocated(dev) / 2 ** 20
    print(f"\n  на карте осталось {left:.0f} МиБ после освобождения "
          f"родительской модели")
    if left > 512:
        raise SystemExit(
            f"родительский процесс всё ещё держит {left:.0f} МиБ: дочерние "
            f"замеры памяти\n  поднимались бы рядом с живой моделью, и "
            f"peak_mib не был бы памятью развёртывания")

    out = dict(ckpt=args.ckpt, script_sha1=k9h.file_sha12(
                   os.path.abspath(__file__)),
               device=str(dev), dtype=args.dtype,
               weight_mode=args.weight_mode, reps=args.reps,
               warmup=args.warmup, passes=PASSES, layers=LAYERS,
               decodes=DECODES, primary_batch=PRIMARY_BATCH,
               protocol=proto, registration=reg,
               orders=[list(o) for o in orders], batches=out_batches)
    out.update(meta)
    out["equivalence"] = out_eq

    # ПАМЯТЬ ПОСЛЕ ВСЕХ ТАЙМИНГОВ. Пока идёт основной процесс, он занимает
    # карту, и дочерний грузил бы модель рядом с ней: чужого allocator это не
    # касается, но соседство влияет на чистоту стенда и может дать OOM.
    n_want = 0 if args.no_mem else len(CONFIGS) * len(batches)
    n_got = 0
    if not args.no_mem:
        print("\n  память: отдельный процесс на руку, после таймингов",
              flush=True)
        import gc as _gc
        _gc.collect()
        torch.cuda.empty_cache()
        for bs in batches:
            print(f"    батч {bs}")
            for name in CONFIGS:
                cmd = [sys.executable, os.path.abspath(__file__),
                       "--mem-only", name, "--ckpt", args.ckpt,
                       "--cache", args.cache, "--joint12", args.joint12,
                       "--hicora-s0", args.hicora_s0,
                       "--hicora-s1", args.hicora_s1, "--root", args.root,
                       "--cfg-path", args.cfg_path, "--device", args.device,
                       "--dtype", args.dtype, "--batches", str(bs),
                       "--pos-offset", str(args.pos_offset),
                       "--weight-mode", args.weight_mode]
                pr = subprocess.run(cmd, capture_output=True, text=True)
                ln = [x for x in pr.stdout.splitlines()
                      if x.startswith("MEMJSON ")]
                if pr.returncode == 0 and ln:
                    v = json.loads(ln[-1][len("MEMJSON "):])["peak_mib"]
                    rows_by_batch[bs][name]["peak_mib"] = v
                    n_got += 1
                    print(f"      {name:<12}{v:>8.0f}М")
                else:
                    tail = (pr.stderr or pr.stdout).strip().splitlines()[-2:]
                    print(f"      {name:<12}  НЕ ИЗМЕРЕНА: {' | '.join(tail)}")
    out["memory_complete"] = bool(args.no_mem or n_got == n_want)
    out["memory_measured"] = n_got
    out["memory_expected"] = n_want

    prim = out["batches"].get(str(PRIMARY_BATCH))
    if prim is None:
        raise SystemExit(f"батч {PRIMARY_BATCH} не измерялся")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                    exist_ok=True)
        json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"\n  сохранено: {args.out} (сырые времена включены)")
    if not out["memory_complete"]:
        print(f"\n  ПАМЯТЬ ИЗМЕРЕНА НЕ ПОЛНОСТЬЮ: {n_got} из {n_want}. "
              f"Прежде такой прогон\n  печатал «НЕ ИЗМЕРЕНА» и всё равно "
              f"завершался успехом. Либо разберитесь\n  с падением "
              f"дочернего процесса, либо запустите с --no-mem осознанно.")
        raise SystemExit(1)
    tag = "" if reg["registered"] else " [НЕ ЗАРЕГИСТРИРОВАННЫЙ ПРОГОН]"
    if not prim["gate"]["passed"]:
        print(f"\n  K-11h-A НЕ ПРОЙДЕН{tag}. Порог зарегистрирован до "
              f"запуска.")
        raise SystemExit(1)
    print(f"\n  K-11h-A ПРОЙДЕН{tag}.")
    print("  ЧТО ЭТО ЗНАЧИТ: преимущество по стоимости при СОПОСТАВИМОМ "
          "НАБЛЮДАЕМОМ успехе.\n  «Не хуже и дешевле» отсюда НЕ следует: "
          "K-11e превосходства не показал,\n  а не-худшесть с допуском мы не "
          "регистрировали — интервалы допускают\n  ухудшение до 1.25-2.5 пп.")


if __name__ == "__main__":
    main()
