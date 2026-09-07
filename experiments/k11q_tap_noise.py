"""K-11q. Откуда 1.9% невязки между кэшированным h24 и живым проходом.

ПОВОД. K-11b (8a5897441056) измерил относительную невязку 1.88e-02 при
контролях 0.96 и 1.38. Контроль пройден с запасом в полсотни раз, то есть
кэш заведомо не подменён. Но 1.9% — это на два порядка больше шума хранения
fp16 (относительная точность около 5e-4), и списывать такую величину на
округление нельзя.

ДВЕ ГИПОТЕЗЫ, КОТОРЫЕ НАДО РАЗДЕЛИТЬ.

  A. Состав батча и набивка. Кэш собран батчами по 16 из СОСЕДНИХ наблюдений
     одного эпизода — длины подсказок близки, набивки мало. Проверка K-11b
     берёт по одному наблюдению из РАЗНЫХ эпизодов и задач, длины подсказок
     разные, набивки много. Тогда 1.9% — цена формы, одинаковая для любых
     двух разных разбиений, и кэш пригоден.

  B. Наблюдение зависит от соседей по батчу. Если `position_ids` при левой
     набивке считаются не так или внимание протекает между примерами, то
     h24 одного и того же наблюдения МЕНЯЕТСЯ от того, с кем оно поехало.
     Тогда кэшированный вход не соответствует никакому одному режиму
     исполнения, и зонд с обучением читают величину, которой у HiCoRA в
     работе не будет.

РЕШАЮЩИЙ ОПЫТ. Одно и то же наблюдение прогоняется трижды: в одиночку, с
набором соседей A и с набором соседей B. Сравниваются ЕГО СОБСТВЕННЫЕ
строки h24.

  - если одиночный, A и B дают между собой ~1e-4, а с кэшем ~1.9% —
    гипотеза A неполна: дело не в составе батча, а в чём-то, отличающем
    сбор от проверки;
  - если A и B расходятся между собой на те же ~1.9% — верна гипотеза B, и
    кэш описывает вход лишь с точностью до состава батча;
  - если одиночный проход совпадает с кэшем гораздо лучше батчевого, значит
    невязку создаёт набивка, и мерить вход надо без неё.

Скрипт НИЧЕГО не записывает в кэш и не трогает отпечатки: он только меряет.

Запуск:
    python3 experiments/k11q_tap_noise.py --selftest

    PYTHONPATH=$HOME/LIBERO MUJOCO_GL=egl \\
    python3 experiments/k11q_tap_noise.py --ckpt <base> \\
        --cache data/k11a_joint12 --joint-ckpt data/k9d_ep3.pt \\
        --device cuda:1
"""

import argparse
import copy
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k11a_build_hicora_cache as k11a  # noqa: E402
import k11b_hicora_identity as k11b  # noqa: E402

TAPS = (12, 18, 24)


def rel_rms(a, b):
    """Относительная невязка в тех же единицах, что в K-11b."""
    x = np.asarray(a, np.float64)
    y = np.asarray(b, np.float64)
    if x.shape != y.shape:
        raise SystemExit(f"формы {x.shape} и {y.shape}")
    den = float(np.sqrt((y ** 2).mean()))
    if den <= 0:
        raise SystemExit("опора нулевая")
    return float(np.sqrt(((x - y) ** 2).mean()) / den)


def read_noise(repeat, repeat_reseed, solo_vs_cache, batch_vs_cache,
               a_vs_b, storage=5e-4, near=3.0):
    """Пре-регистрированное чтение. ПЕРВЫЙ вопрос — воспроизводим ли проход.

    ЧЕГО НЕ ХВАТАЛО В ПЕРВОЙ ВЕРСИИ. Правило сравнивало влияние соседей с
    расхождением против кэша, но не имело НУЛЕВОГО КОНТРОЛЯ: повторить тот
    же самый вызов и посмотреть, совпадёт ли он сам с собой. Без этого
    вариант «любые два прохода различаются на два процента независимо ни от
    чего» неотличим от «виноваты соседи», и первая версия выдала вердикт B
    на данных, где `одиночно/А` был такой же величины, как всё остальное, —
    то есть на данных, которые B как раз НЕ подтверждают.

    Порядок вопросов теперь такой:
      1. повтор того же вызова — если расходится, всё остальное бессмысленно;
      2. повтор после сброса сида — отделяет случайность подготовки входа
         (аугментация картинки и подобное) от недетерминизма ядер;
      3. только потом — соседи, набивка, округление.
    """
    if repeat > storage * near:
        if repeat_reseed <= storage * near:
            return "rng", (
                f"СЛУЧАЙНОСТЬ ПОДГОТОВКИ ВХОДА: два одинаковых вызова "
                f"расходятся на {repeat:.2e}, но после сброса сида — лишь на "
                f"{repeat_reseed:.2e}. Значит вход зависит от состояния "
                f"генератора: где-то в подготовке (скорее всего в обработке "
                f"картинки) есть случайное преобразование. Кэш описывает ОДИН "
                f"розыгрыш, а в работе будет другой — и это касается не "
                f"только HiCoRA, но и всех прошлых измерений на этом кэше")
            
        return "nondet", (
            f"НЕДЕТЕРМИНИЗМ ПРОХОДА: два одинаковых вызова расходятся на "
            f"{repeat:.2e}, и сброс сида не помогает ({repeat_reseed:.2e}). "
            f"Величина того же порядка, что расхождение с кэшем "
            f"({batch_vs_cache:.2e}), поэтому вопрос о соседях и набивке "
            f"поставлен неверно: воспроизводимости нет на уровне одного "
            f"вызова. Искать в ядрах внимания и autocast")
    if a_vs_b > batch_vs_cache / near:
        return "B", (
            f"ЗАВИСИМОСТЬ ОТ СОСЕДЕЙ: повтор воспроизводим ({repeat:.2e}), "
            f"но одно и то же наблюдение с разными соседями даёт "
            f"{a_vs_b:.2e} при расхождении с кэшем {batch_vs_cache:.2e}. "
            f"Кэшированный вход определён лишь с точностью до состава батча")
    if solo_vs_cache < batch_vs_cache / near:
        return "padding", (
            f"НАБИВКА: повтор воспроизводим ({repeat:.2e}), одиночный проход "
            f"совпадает с кэшем на {solo_vs_cache:.2e}, батчевый лишь на "
            f"{batch_vs_cache:.2e}. Невязку создаёт набивка разнородных "
            f"подсказок, а не кэш")
    if solo_vs_cache <= storage * near:
        return "storage", (
            f"ХРАНЕНИЕ: повтор воспроизводим ({repeat:.2e}), одиночный проход "
            f"совпадает с кэшем на {solo_vs_cache:.2e} — уровень округления "
            f"fp16. Кэш верен")
    return "unexplained", (
        f"НЕ ОБЪЯСНЕНО: повтор воспроизводим ({repeat:.2e}), соседи влияют "
        f"на {a_vs_b:.2e}, а одиночный проход расходится с кэшем на "
        f"{solo_vs_cache:.2e} при шуме хранения {storage:.0e}. Различие "
        f"между сбором и проверкой есть, но оно не в составе батча и не в "
        f"округлении — искать в подготовке входа: состояние, картинка, "
        f"подсказка, офсет")


def read_mechanism(interv_vs_solo, interv_vs_batch, batch_effect, near=3.0):
    """Причина устанавливается ИНТЕРВЕНЦИЕЙ, а не совпадением изменений.

    Прежняя версия объявляла механизм по тому, что позиции у одного и того
    же наблюдения различаются между батчами. Это корреляция: позиции меняются
    ОДНОВРЕМЕННО с составом батча, длиной набивки и всем прочим.

    Здесь подменены только позиции, при тех же вложениях, маске и форме.
    `interv_vs_solo` — насколько это сдвинуло результат от одиночного;
    `interv_vs_batch` — сколько осталось до батчевого.

    ЧЕТЫРЕ ИСХОДА, И ТРЕТИЙ ЛЕГКО СПУТАТЬ СО ВТОРЫМ. Собственный эффект
    позиций и объяснение ими эффекта батча — РАЗНЫЕ вещи: подмена может
    сдвигать результат ровно на ту же величину и при этом не приближать к
    батчевому ни на сколько. Тогда позиции причинны, но разрыв объясняется
    не ими.
    """
    close = batch_effect / near
    moved = interv_vs_solo >= close
    arrived = interv_vs_batch <= close
    if moved and arrived:
        return ("МЕХАНИЗМ — ПОЗИЦИИ: подмена ТОЛЬКО position_ids сдвигает "
                f"результат на {interv_vs_solo:.2e} и приводит его к "
                f"батчевому с точностью {interv_vs_batch:.2e}. Чинится "
                "однородными по длине батчами либо правкой построения позиций")
    if not moved:
        return ("МЕХАНИЗМ НЕ В ПОЗИЦИЯХ: их подмена почти ничего не меняет "
                f"({interv_vs_solo:.2e} против батчевого эффекта "
                f"{batch_effect:.2e}). Совпадение изменения позиций с "
                "изменением h24 оказалось корреляцией. Искать в набивке, "
                "маске и форме батча")
    if interv_vs_batch >= batch_effect * 0.8:
        return ("ПОЗИЦИИ ПРИЧИННЫ, НО РАЗРЫВ ОБЪЯСНЯЮТ НЕ ОНИ: их подмена "
                f"сдвигает результат на {interv_vs_solo:.2e}, то есть эффект "
                f"настоящий, но до батчевого остаётся {interv_vs_batch:.2e} "
                f"при исходном разрыве {batch_effect:.2e} — не ближе, чем "
                "было. Значит есть ВТОРАЯ причина сопоставимой величины, и "
                "её природа решает, шум это или загрязнение")
    return ("ПОЗИЦИИ ОБЪЯСНЯЮТ ЧАСТЬ: их подмена сдвигает результат на "
            f"{interv_vs_solo:.2e}, до батчевого остаётся "
            f"{interv_vs_batch:.2e} при исходном разрыве {batch_effect:.2e}. "
            "Причина составная, и одними позициями не исчерпывается")


def read_leak(same_len_diff, batch_effect, ctrl=None, storage=5e-4,
              near=3.0, ctrl_min=50.0):
    """Шум формы или ЗАГРЯЗНЕНИЕ содержимым соседей — это разные вещи.

    ЗАЧЕМ. Если h24 целевой строки меняется от того, ЧТО написано у соседей
    при ОДИНАКОВОЙ длине после набивки, одинаковых позициях и одинаковой
    форме, значит содержание чужих примеров протекает в целевую строку.
    Тогда состав батча способен не только ослабить сигнал, но и создать
    ложный межгрупповой: соседи коррелируют с задачей и с порядком сбора.

    Если при одинаковой длине результат не меняется, остаётся нуисанс,
    зависящий только от длин и формы батча. Он добавляет шум, но чужого
    содержания не несёт.
    """
    # КОНТРОЛЬ ПЕРВЫМ. Ровный ноль у целевой строки означает «протекания
    # нет» только если соседи ДЕЙСТВИТЕЛЬНО различались. Если строка самого
    # соседа тоже не изменилась, сравнивались одинаковые батчи, и вывод
    # делать не о чем.
    if ctrl is not None and ctrl <= storage * ctrl_min:
        return "void", (
            f"ПРОВЕРКА НЕДЕЙСТВИТЕЛЬНА: строка самого соседа изменилась лишь "
            f"на {ctrl:.2e}, то есть два батча были практически одинаковы. "
            f"Ровный ноль у целевой строки ({same_len_diff:.2e}) в таком "
            f"случае ничего не означает — нужны соседи заведомо разного "
            f"содержания при той же длине")
    if same_len_diff > storage * near:
        return "leak", (
            f"ЗАГРЯЗНЕНИЕ: при ОДИНАКОВОЙ длине после набивки и одинаковых "
            f"позициях смена содержимого соседей меняет h24 целевой строки "
            f"на {same_len_diff:.2e}. Содержание чужих примеров протекает в "
            f"целевую. Состав батча тогда способен создать ЛОЖНЫЙ сигнал, а "
            f"не только ослабить настоящий, и выводы зонда требуют проверки "
            f"в нескольких контролируемых контекстах")
    return "shape", (
        (f"ЗАГРЯЗНЕНИЯ НЕТ (контроль: сам сосед изменился на {ctrl:.2e}). "
         if ctrl is not None else "ЗАГРЯЗНЕНИЯ НЕТ: ")
        + f"При одинаковой длине после набивки смена "
        f"содержимого соседей меняет h24 лишь на {same_len_diff:.2e} — на "
        f"уровне округления. Изменчивость зависит от ДЛИН и формы батча, а "
        f"не от чужого содержания: это нуисанс, коррелированный с длиной "
        f"подсказки, но чужих примеров в нём нет")


def selftest():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 16, 32))
    assert rel_rms(x, x) == 0.0
    # величина относительная: общий масштаб не влияет
    y = x + 0.01 * rng.normal(size=x.shape)
    assert abs(rel_rms(x, y) - rel_rms(10 * x, 10 * y)) < 1e-12
    # и растёт с расхождением
    assert rel_rms(x + 0.1, x) > rel_rms(x + 0.01, x)
    try:
        rel_rms(x[:2], x)
    except SystemExit:
        pass
    else:
        raise AssertionError("разные формы приняты")

    # --- чтение вердикта ----------------------------------------------------
    # НУЛЕВОЙ КОНТРОЛЬ ПЕРВЫМ: невоспроизводимый повтор обесценивает всё
    # остальное, и правило обязано это говорить, а не рассуждать о соседях.
    tag, txt = read_noise(repeat=2.0e-2, repeat_reseed=1e-5,
                          solo_vs_cache=1.8e-2, batch_vs_cache=1.9e-2,
                          a_vs_b=1.9e-2)
    assert tag == "rng" and "генератора" in txt, tag
    tag, txt = read_noise(2.0e-2, 2.0e-2, 1.8e-2, 1.9e-2, 1.9e-2)
    assert tag == "nondet" and "воспроизводимости нет" in txt, tag
    # ИМЕННО ТА КАРТИНА, на которой прежнее правило выдало B: все пары
    # одинаковы. Теперь она обязана читаться как отсутствие воспроизводимости,
    # а не как влияние соседей.
    assert read_noise(1.9e-2, 1.9e-2, 1.77e-2, 1.91e-2, 1.89e-2)[0] != "B"
    # при воспроизводимом повторе прежние исходы сохраняются
    tag, _ = read_noise(1e-5, 1e-5, 1.8e-2, 1.9e-2, 1.9e-2)
    assert tag == "B", tag
    tag, _ = read_noise(1e-5, 1e-5, 1e-4, 1.9e-2, 1e-4)
    assert tag in ("padding", "storage"), tag
    tag, _ = read_noise(1e-5, 1e-5, 3e-4, 3.5e-4, 1e-5)
    assert tag == "storage", tag
    tag, txt = read_noise(1e-5, 1e-5, 1.8e-2, 1.9e-2, 1e-5)
    assert tag == "unexplained" and "подготовке входа" in txt, tag
    # КОНТРОЛЬ: «хранение» не выдаётся при невязке в проценты
    for s_ in (1e-2, 5e-3):
        assert read_noise(1e-5, 1e-5, s_, s_ * 1.05, 1e-6)[0] != "storage"

    # --- механизм устанавливается интервенцией ------------------------------
    # позиции сдвигают к батчевому и почти доводят до него -> они и есть
    t_ = read_mechanism(1.9e-2, 1e-3, 1.9e-2)
    assert t_.startswith("МЕХАНИЗМ — ПОЗИЦИИ"), t_
    # подмена позиций почти ничего не меняет -> корреляция, а не причина
    t_ = read_mechanism(1e-4, 1.9e-2, 1.9e-2)
    assert "НЕ В ПОЗИЦИЯХ" in t_ and "корреляцией" in t_, t_
    # ИМЕННО ИЗМЕРЕННАЯ КАРТИНА: подмена сдвигает ровно на величину эффекта,
    # но не приближает к батчевому. Это НЕ «объясняют часть».
    t_ = read_mechanism(1.93e-2, 1.93e-2, 1.91e-2)
    assert "РАЗРЫВ ОБЪЯСНЯЮТ НЕ ОНИ" in t_ and "ВТОРАЯ причина" in t_, t_
    assert "ЧАСТЬ" not in t_, t_
    # а вот это действительно частичное объяснение
    t_ = read_mechanism(1.0e-2, 1.0e-2, 1.9e-2)
    assert "ЧАСТЬ" in t_, t_
    # КОНТРОЛЬ: вердикт «позиции» не выдаётся, когда до батчевого далеко
    assert not read_mechanism(1.9e-2, 1.8e-2, 1.9e-2).startswith(
        "МЕХАНИЗМ — ПОЗИЦИИ")

    # --- загрязнение отличается от шума формы -------------------------------
    tag_, txt_ = read_leak(1.5e-2, 1.9e-2, ctrl=0.5)
    assert tag_ == "leak" and "ЛОЖНЫЙ" in txt_, txt_
    tag_, txt_ = read_leak(2e-4, 1.9e-2, ctrl=0.5)
    assert tag_ == "shape" and "ЗАГРЯЗНЕНИЯ НЕТ" in txt_, txt_
    # «загрязнения нет» не выдаётся при расхождении в проценты
    for v_ in (1e-2, 5e-3, 2e-3):
        assert read_leak(v_, 1.9e-2, ctrl=0.5)[0] == "leak", v_
    # ГЛАВНЫЙ КОНТРОЛЬ: если сам сосед не изменился, сравнивались одинаковые
    # батчи, и ровный ноль у цели не значит ничего. Без этой ветки «0.00»
    # читался бы как доказательство отсутствия протекания.
    tag_, txt_ = read_leak(0.0, 1.9e-2, ctrl=0.0)
    assert tag_ == "void" and "НЕДЕЙСТВИТЕЛЬНА" in txt_, txt_
    assert read_leak(0.0, 1.9e-2, ctrl=1e-3)[0] == "void"
    # а при живом контроле тот же ноль читается как отсутствие протекания
    assert read_leak(0.0, 1.9e-2, ctrl=0.3)[0] == "shape"
    # без контроля вовсе — прежнее поведение сохраняется
    assert read_leak(0.0, 1.9e-2)[0] == "shape"

    print("самопроверка k11q пройдена (версия «интервенция и проверка загрязнения»): "
          "относительная невязка не зависит от масштаба и растёт с "
          "расхождением, чтение СНАЧАЛА требует воспроизводимости повтора и "
          "на картине из первого прогона больше НЕ выдаёт вердикт о соседях, "
          "отделяет случайность подготовки входа от недетерминизма ядер, "
          "при воспроизводимом повторе различает соседей, набивку, "
          "округление и необъяснённый остаток, механизм устанавливает "
          "ИНТЕРВЕНЦИЕЙ по позициям и не выдаёт корреляцию за причину, "
          "отличает собственный эффект позиций от объяснения ими разрыва, а "
          "загрязнение содержимым соседей — от шума формы батча, объявляя "
          "проверку недействительной, если сам сосед не изменился")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--cache")
    ap.add_argument("--joint-ckpt")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--n-comp", type=int, default=7,
                    help="соседей в каждом из двух наборов")
    ap.add_argument("--n-target", type=int, default=3,
                    help="сколько разных наблюдений проверить")
    ap.add_argument("--out", default="data/k11q_tap_noise.json")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    selftest()

    for f in ("cache", "joint_ckpt", "out"):
        if getattr(args, f):
            setattr(args, f, os.path.abspath(getattr(args, f)))
    args.root = os.path.abspath(args.root)
    sys.path.insert(0, args.root)
    print(f"k11q sha1 {k11a.file_sha1(__file__)}")
    for need, why in ((args.ckpt, "--ckpt"), (args.cache, "--cache"),
                      (args.joint_ckpt, "--joint-ckpt")):
        if not need:
            raise SystemExit(f"нужен {why}")

    import torch
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    import actioncodec  # noqa: F401
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    if "action_codec" not in CONFIG_MAPPING:
        raise SystemExit("тип «action_codec» не зарегистрирован")
    from smolvla.bar import SmolVLABlockwiseAR
    from utils import (STATE_Q01, STATE_Q99, VisionLanguageActionProcessor,
                       dict_apply, get_cfg, process_state, prompt_template,
                       seed_everything)
    from joint12_vla import make_joint12_class
    import hicora_vla as hv

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("контейнер потерял видеокарты")
    seed_everything(0)
    dev, dt = torch.device(args.device), getattr(torch, args.dtype)
    cfg = get_cfg(os.path.join(args.root, args.cfg_path))
    cfg.TRAINING.ckpt_dir = args.ckpt
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = args.ckpt

    meta = json.load(open(args.cache + ".meta.json"))
    ds_repo, ds_rev = k11b.dataset_source(meta)
    tap = max(meta["saved_taps"])

    Cls = make_joint12_class(SmolVLABlockwiseAR)
    model = Cls.from_pretrained(**cfg.MODEL.vlm.kwargs).to(dev, dt).eval()
    proc = VisionLanguageActionProcessor.from_pretrained(
        args.ckpt, trust_remote_code=True, mode="discrete")
    model.init_joint_fast(depth=args.depth, head_dtype=dt)
    res_norm = copy.deepcopy(model.action_expert.norm)
    # ЗАГРУЗКА ДОСЛОВНО КАК В K-11a. Веса Joint12 перезаписывают
    # `bos_embedding`, участвующий во всех 24 шагах внимания, поэтому h24 от
    # них зависит напрямую: частично или иначе применённый чекпойнт дал бы
    # правдоподобные, но НЕ СОПОСТАВИМЫЕ с кэшем числа, а весь смысл этого
    # скрипта — в сопоставимости.
    wsha = k11a.file_sha1(args.joint_ckpt)
    src_sha = (meta.get("source") or {}).get("weights_sha1")
    if src_sha != wsha:
        raise SystemExit(f"кэш собран весами sha {src_sha}, а здесь {wsha}")
    obj = torch.load(args.joint_ckpt, map_location="cpu", weights_only=False)
    if int(obj.get("depth", -1)) != args.depth:
        raise SystemExit(f"чекпойнт глубины {obj.get('depth')}, задано "
                         f"{args.depth}")
    state = obj["state"]
    own = dict(model.named_parameters())
    stray = [k for k in state
             if not any(k.startswith(pp) or k == pp.rstrip(".")
                        for pp in model.trainable_prefixes)]
    missing = [k for k in own if own[k].requires_grad and k not in state]
    if stray or missing:
        raise SystemExit(f"ключей вне белого списка {len(stray)}, "
                         f"недостающих обучаемых {len(missing)}")
    with torch.no_grad():
        for k, v in state.items():
            if tuple(own[k].shape) != tuple(v.shape):
                raise SystemExit(f"форма {k}")
            own[k].data = v.to(dev, torch.float32)
    model.to_fp32_trainable()
    not32 = [k for k in state if own[k].dtype != torch.float32]
    if not32:
        raise SystemExit(f"{len(not32)} загруженных весов не в fp32")
    print(f"  веса Joint12 sha {wsha} совпали с кэшем, {len(state)} тензоров "
          f"в fp32")
    model.eval()
    model.__class__ = hv.make_hicora_class(type(model))
    # КОДЕК ЗДЕСЬ НЕ НУЖЕН И НЕ ПОДНИМАЕТСЯ. Скрипт делает только
    # `forward_taps` и сравнивает отводы; ничего не декодируется, кодовые
    # книги ни во что не входят. Прежняя версия доставала их и падала на
    # том, что квантователи остались на CPU, — лишняя зависимость, которая
    # могла только сломаться.
    model.set_res_norm(res_norm.to(dev))
    model.taps, model.q0_depth = TAPS, args.depth
    model.n_layers_total = len(model.action_expert.layers)

    d = np.load(meta["cache"], allow_pickle=True)
    IMG = np.load(meta["cache"] + ".images.npy", mmap_mode="r")
    offs = d["pos_offset"].astype(np.int64)
    epi, stp, tsk = d["episode"], d["step"], d["task"]
    H = np.load(f"{args.cache}.h{tap}.npy", mmap_mode="r")

    def states_for(idxs):
        st = np.zeros((len(idxs), len(STATE_Q01)), np.float64)
        for j, gi in enumerate(idxs):
            e = int(epi[gi])
            f = hf_hub_download(
                ds_repo, f"data/chunk-{e // 1000:03d}/episode_{e:06d}.parquet",
                repo_type="dataset", revision=ds_rev)
            S_ = np.asarray(pq.read_table(f).column("state").to_pylist(),
                            np.float32)
            st[j] = S_[int(stp[gi])] if S_.shape[1] == len(STATE_Q01) \
                else process_state(S_[int(stp[gi])][None])[0]
        return (st - STATE_Q01) / (STATE_Q99 - STATE_Q01) * 2.0 - 1.0

    def run(sel, po, pos_override=None):
        """h24 для набора наблюдений и длины подсказок до набивки.

        `pos_override` подменяет `position_ids` целиком, не трогая ничего
        другого: это и есть интервенция, отделяющая позиции от прочего.
        """
        st_n = states_for(sel)
        image = torch.from_numpy(np.asarray(IMG[sel]))
        msgs = []
        for j, gi in enumerate(sel):
            m = prompt_template(
                st_n[j], None, str(tsk[gi]),
                mode=cfg.MODEL.vla_processor.kwargs.mode,
                action_vocab_size=cfg.MODEL.action_processor.vocab_size,
                action_token_len=cfg.MODEL.action_processor.token_len)
            m[1]["content"] = m[1]["content"][1:]
            msgs.append(m)
        texts = proc.apply_chat_template(msgs, add_generation_prompt=True)
        b = proc(text=texts, images=[[image[i].numpy()]
                                     for i in range(len(sel))],
                 return_tensors="pt", padding=True, padding_side="left",
                 action_processor_kwargs={"embodiment_ids": 0})
        b = dict_apply(lambda x: x.to(dev, dt), b)
        am = b.get("attention_mask")
        lens = (am.sum(-1).detach().cpu().numpy().tolist()
                if am is not None else None)
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=dt):
            v, pp = model.build_inputs(position_offset=int(po), **b)
            # ИНТЕРВЕНЦИЯ: позиции можно подменить, оставив вложения, маску и
            # форму батча теми же. Только так отделяется вклад позиций от
            # вклада всего остального; совпадение изменений причиной не
            # является, тем более что при точной арифметике одинаковый сдвиг
            # всех валидных позиций относительные углы RoPE сохраняет.
            if pos_override is not None:
                pp = pos_override.to(pp.device, pp.dtype)
            tp = model.forward_taps(vlm_inputs_embeds=v, attention_mask=am,
                                    position_ids=pp)
        # ПОЗИЦИИ И ВЛОЖЕНИЯ ЦЕЛЕВОЙ СТРОКИ БЕЗ НАБИВКИ. Если они у одного и
        # того же наблюдения зависят от батча, механизм найден: RoPE считает
        # позицию от начала НАБИТОЙ строки, а не от начала подсказки.
        #
        # ДЛИНЫ РАЗНЫЕ. `position_ids` покрывают подсказку И поток действий
        # (179 + 16 = 195), а маска внимания — только подсказку. Поток
        # действий и есть то, откуда снимается h24, поэтому его позиции
        # выделяются отдельно и сравниваются тоже.
        def unpad(x):
            if am is None:
                return x[0].detach().float().cpu().numpy()
            L = int(am.shape[-1])
            row, m = x[0], (am[0] == 1)
            if row.shape[0] == L:
                return row[m].detach().float().cpu().numpy()
            if row.shape[0] > L:
                head = row[:L][m]
                return torch.cat([head, row[L:]], 0).detach().float(
                    ).cpu().numpy()
            raise SystemExit(f"длина {row.shape[0]} меньше маски {L}")

        n_act = int(pp.shape[-1]) - int(am.shape[-1]) if am is not None else 0
        pos_all = unpad(pp if pp.dim() > 1 else pp.unsqueeze(0))
        pos_act = pos_all[-n_act:] if n_act > 0 else pos_all[-16:]
        return (tp[tap].float().cpu().numpy().astype(np.float16), lens,
                pos_all, unpad(v), pos_act, pp.detach().clone())

    # --- выбор: одна цель, два непересекающихся набора соседей -------------
    rng = np.random.default_rng(3)
    po = int(rng.choice(sorted({int(v) for v in offs})))
    cand = np.where(offs == po)[0]
    eps = np.unique(epi[cand])
    if len(eps) < 2 * args.n_comp + args.n_target:
        raise SystemExit("мало эпизодов на этом офсете")
    order = rng.permutation(eps)
    tgt_eps = order[:args.n_target]
    comp_a = order[args.n_target:args.n_target + args.n_comp]
    comp_b = order[args.n_target + args.n_comp:
                   args.n_target + 2 * args.n_comp]

    def one_per_ep(ep_list):
        out = []
        for e in ep_list:
            same = cand[epi[cand] == e]
            out.append(int(same[rng.integers(0, len(same))]))
        return np.asarray(out)

    A = one_per_ep(comp_a)
    B = one_per_ep(comp_b)
    print(f"  офсет {po}, целей {args.n_target}, соседей по {args.n_comp} "
          f"в двух непересекающихся наборах, все из разных эпизодов")

    # НУЛЕВОЙ КОНТРОЛЬ ДО ВСЕГО ОСТАЛЬНОГО: тот же самый вызов дважды, и он
    # же с одинаково сброшенным сидом. Без этого любые выводы о соседях и
    # набивке повисают в воздухе.
    # Берём уже выбранные наблюдения, не трогая генератор: лишний розыгрыш
    # сдвинул бы состав целей и прогон перестал бы сравниваться с прошлым.
    probe_sel = np.concatenate([[int(B[0])], A])
    r1, *_ = run(probe_sel, po)
    r2, *_ = run(probe_sel, po)
    rep = rel_rms(r1, r2)
    seed_everything(0)
    r3, *_ = run(probe_sel, po)
    seed_everything(0)
    r4, *_ = run(probe_sel, po)
    rep_seed = rel_rms(r3, r4)
    print(f"  ПОВТОР того же вызова: {rep:.2e}; он же со сбросом сида перед "
          f"каждым: {rep_seed:.2e}")

    rows = []
    for t in one_per_ep(tgt_eps):
        cache_t = np.asarray(H[[t]])[0]
        solo, len_solo, pos_s, emb_s, act_s, raw_s = run(np.asarray([t]), po)
        ha, len_a, pos_a, emb_a, act_a, raw_a = run(
            np.concatenate([[t], A]), po)
        hb, len_b, pos_b, emb_b, act_b, _ = run(np.concatenate([[t], B]), po)
        # ИНТЕРВЕНЦИЯ ПО ПОЗИЦИЯМ. Тот же одиночный вход, та же маска, та же
        # форма — подменены ТОЛЬКО позиции на те, что наблюдение получило бы
        # в батче А. Если h24 после этого совпадёт с батчевым, позиции и есть
        # причина; если останется на месте — причина в другом.
        n_solo = raw_s.shape[-1]
        pos_from_batch = raw_a[:1, -n_solo:] if raw_a.dim() > 1 \
            else raw_a[-n_solo:].unsqueeze(0)
        solo_pi, *_ = run(np.asarray([t]), po, pos_override=pos_from_batch)
        iv_vs_solo = rel_rms(solo_pi[0], solo[0])
        iv_vs_batch = rel_rms(solo_pi[0], ha[0])
        # ТО ЖЕ ПОСЛЕ НОРМЫ. Голова читает `res_norm(h24)`, а не сырой отвод;
        # норма может расхождение сжать или усилить, и мерить надо тот вход,
        # который HiCoRA действительно получает.
        def nrm(x):
            with torch.no_grad():
                return model.res_norm(
                    torch.from_numpy(x).to(dev, dt)).float().cpu().numpy()
        n_solo_, n_pi_, n_ha_ = nrm(solo), nrm(solo_pi), nrm(ha)
        ivn_vs_solo = rel_rms(n_pi_[0], n_solo_[0])
        ivn_vs_batch = rel_rms(n_pi_[0], n_ha_[0])
        eff_solo_a = rel_rms(solo[0], ha[0])
        eff_norm_a = rel_rms(n_solo_[0], n_ha_[0])
        # МЕХАНИЗМ: совпадают ли позиции и вложения целевой строки
        same_pos = (pos_s is not None and pos_a is not None
                    and pos_s.shape == pos_a.shape
                    and bool(np.array_equal(pos_s, pos_a))
                    and pos_a.shape == pos_b.shape
                    and bool(np.array_equal(pos_a, pos_b)))
        emb_d = (float(np.abs(emb_s - emb_a).max())
                 if emb_s is not None and emb_a is not None
                 and emb_s.shape == emb_a.shape else None)
        r = dict(
            idx=int(t), episode=int(epi[t]), task=str(tsk[t]),
            solo_vs_cache=rel_rms(solo[0], cache_t),
            batchA_vs_cache=rel_rms(ha[0], cache_t),
            batchB_vs_cache=rel_rms(hb[0], cache_t),
            A_vs_B=rel_rms(ha[0], hb[0]),
            solo_vs_A=rel_rms(solo[0], ha[0]),
            len_solo=len_solo, len_A=len_a, len_B=len_b,
            same_position_ids=same_pos,
            same_action_positions=bool(
                act_s.shape == act_a.shape and np.array_equal(act_s, act_a)
                and act_a.shape == act_b.shape
                and np.array_equal(act_a, act_b)),
            act_pos_solo=act_s.tolist()[:6], act_pos_A=act_a.tolist()[:6],
            pos_solo_head=(pos_s[:4].tolist() if pos_s is not None else None),
            pos_A_head=(pos_a[:4].tolist() if pos_a is not None else None),
            emb_max_abs_diff=emb_d,
            interv_pos_vs_solo=iv_vs_solo, interv_pos_vs_batch=iv_vs_batch,
            interv_norm_vs_solo=ivn_vs_solo,
            interv_norm_vs_batch=ivn_vs_batch,
            effect_solo_vs_A=eff_solo_a, effect_norm_solo_vs_A=eff_norm_a)
        rows.append(r)
        print(f"    набл. {t} (эпизод {r['episode']}): одиночно/кэш "
              f"{r['solo_vs_cache']:.2e}, батчА/кэш {r['batchA_vs_cache']:.2e}, "
              f"батчБ/кэш {r['batchB_vs_cache']:.2e}, А/Б {r['A_vs_B']:.2e}, "
              f"одиночно/А {r['solo_vs_A']:.2e}")
        print(f"      длины без набивки: одиночно {len_solo}, "
              f"в А {sorted(set(len_a))[:4]}..., в Б {sorted(set(len_b))[:4]}...")
        print(f"      позиции подсказки совпадают: {same_pos}; позиции "
              f"ПОТОКА ДЕЙСТВИЙ совпадают: {r['same_action_positions']}")
        print(f"      ИНТЕРВЕНЦИЯ (подменены только позиции): сдвиг от "
              f"одиночного {iv_vs_solo:.2e}, остаток до батчевого "
              f"{iv_vs_batch:.2e}, сам эффект {eff_solo_a:.2e}")
        print(f"      она же ПОСЛЕ res_norm: сдвиг {ivn_vs_solo:.2e}, "
              f"остаток {ivn_vs_batch:.2e}, эффект {eff_norm_a:.2e}")
        print(f"      действия одиночно {r['act_pos_solo']}, в А "
              f"{r['act_pos_A']}; вложения расходятся на "
              + ("—" if emb_d is None else f"{emb_d:.2e}"))

    # --- ЗАГРЯЗНЕНИЕ ИЛИ ШУМ ФОРМЫ ------------------------------------------
    # Два батча ОДИНАКОВОЙ длины после набивки: позиции и форма совпадают,
    # различается только СОДЕРЖИМОЕ соседей. Если h24 целевой строки при этом
    # меняется, содержание чужих примеров протекает в целевую.
    leak_diff, leak_ctrl, leak_note = None, None, "не выполнено"
    pool_le = one_per_ep(order[args.n_target + 2 * args.n_comp:][:24])
    if len(pool_le) >= 2:
        _, lens_le, *_ = run(pool_le, po)
        by_len = {}
        for gi, L in zip(pool_le, lens_le or []):
            by_len.setdefault(int(L), []).append(int(gi))
        pair = next((v for v in by_len.values() if len(v) >= 2), None)
        if pair is None:
            leak_note = ("не нашлось двух соседей одинаковой длины — "
                         "проверка загрязнения пропущена")
        else:
            t_le = int(one_per_ep(tgt_eps[:1])[0])
            c_, d_ = pair[0], pair[1]
            h_c, ln_c, *_ = run(np.asarray([t_le] + [c_] * args.n_comp), po)
            h_d, ln_d, *_ = run(np.asarray([t_le] + [d_] * args.n_comp), po)
            if sorted(set(ln_c)) != sorted(set(ln_d)):
                leak_note = (f"длины после набивки разошлись ({set(ln_c)} "
                             f"против {set(ln_d)}) — проверка недействительна")
            else:
                leak_diff = rel_rms(h_c[0], h_d[0])
                # КОНТРОЛЬ: строка САМОГО СОСЕДА обязана различаться. Без
                # него ровный ноль у целевой строки означал бы и «протекания
                # нет», и «я сравнил два одинаковых батча»: соседи равной
                # длины могли совпасть токен в токен, если задача та же, а
                # состояния дискретизовались одинаково.
                leak_ctrl = rel_rms(h_c[1], h_d[1])
                print(f"\n  загрязнение: соседи c={c_} и d={d_} одной длины, "
                      f"строки батча {sorted(set(ln_c))}; целевая строка "
                      f"меняется на {leak_diff:.2e}, сам сосед — на "
                      f"{leak_ctrl:.2e} (контроль)")
    else:
        leak_note = "мало эпизодов для проверки загрязнения"

    med = lambda k: float(np.median([r[k] for r in rows]))
    tag, verdict = read_noise(rep, rep_seed, med("solo_vs_cache"),
                              med("batchA_vs_cache"), med("A_vs_B"))
    print(f"\n  медианы: повтор {rep:.2e}, повтор с сидом {rep_seed:.2e}, "
          f"одиночно/кэш {med('solo_vs_cache'):.2e}, "
          f"батч/кэш {med('batchA_vs_cache'):.2e}, "
          f"А/Б {med('A_vs_B'):.2e}, одиночно/А {med('solo_vs_A'):.2e}")
    n_same = sum(1 for r in rows if r["same_position_ids"])
    n_act = sum(1 for r in rows if r["same_action_positions"])
    print(f"\n  позиции подсказки совпали у {n_same} из {len(rows)}, позиции "
          f"потока действий — у {n_act} из {len(rows)}")
    iv_s = float(np.median([r["interv_pos_vs_solo"] for r in rows]))
    iv_b = float(np.median([r["interv_pos_vs_batch"] for r in rows]))
    ivn_s = float(np.median([r["interv_norm_vs_solo"] for r in rows]))
    ivn_b = float(np.median([r["interv_norm_vs_batch"] for r in rows]))
    # ОПОРА — ТОТ ЖЕ ЭФФЕКТ, КОТОРЫЙ ОБЪЯСНЯЕМ. Прежде порог нормировался на
    # `batchA_vs_cache` — расхождение батча с КЭШЕМ, тогда как интервенция
    # сравнивает одиночный проход с батчем. Величины близки, но это разные
    # опоры, и правило должно нормироваться на ту, что объясняет.
    eff_raw = float(np.median([r["effect_solo_vs_A"] for r in rows]))
    eff_norm = float(np.median([r["effect_norm_solo_vs_A"] for r in rows]))
    print(f"  интервенция по позициям (сырой отвод): сдвиг {iv_s:.2e}, "
          f"остаток {iv_b:.2e}, объясняемый эффект {eff_raw:.2e}")
    print(f"  она же после res_norm: сдвиг {ivn_s:.2e}, остаток "
          f"{ivn_b:.2e}, эффект {eff_norm:.2e}")
    if tag == "B":
        verdict += (".\n  СЫРОЙ ОТВОД: "
                    + read_mechanism(iv_s, iv_b, eff_raw)
                    + ".\n  ВХОД ГОЛОВЫ (после res_norm): "
                    + read_mechanism(ivn_s, ivn_b, eff_norm))
        if leak_diff is not None:
            leak_tag, leak_txt = read_leak(leak_diff, med("batchA_vs_cache"),
                                           ctrl=leak_ctrl)
            verdict += ".\n  " + leak_txt
        else:
            leak_tag = None
            verdict += f".\n  Проверка загрязнения: {leak_note}"
    else:
        leak_tag = None
    print(f"\n  {verdict}")
    print("  ЧИТАТЬ ТАК: это диагностика ВХОДА, а не тождества и не "
          "качества.\n  Ни один исход не отменяет K-11b: контроль подмены "
          "там пройден с запасом\n  в полсотни раз, речь только о том, "
          "насколько точно кэш описывает вход.")

    out = dict(script_sha1=k11a.file_sha1(__file__), cache=args.cache,
               tap=int(tap), offset=po, rows=rows,
               repeat=rep, repeat_reseed=rep_seed,
               medians={k: med(k) for k in
                        ("solo_vs_cache", "batchA_vs_cache",
                         "batchB_vs_cache", "A_vs_B", "solo_vs_A")},
               leak=dict(diff=leak_diff, ctrl=leak_ctrl, note=leak_note,
                         tag=leak_tag),
               intervention=dict(
                   pos_only_vs_solo=iv_s, pos_only_vs_batch=iv_b,
                   effect_solo_vs_batch=eff_raw,
                   norm_pos_only_vs_solo=ivn_s,
                   norm_pos_only_vs_batch=ivn_b,
                   norm_effect_solo_vs_batch=eff_norm),
               tag=tag, verdict=verdict)
    tmp = args.out + ".tmp"
    json.dump(out, open(tmp, "w"), ensure_ascii=False, indent=1)
    os.replace(tmp, args.out)
    print(f"\n  сохранено: {args.out}")


if __name__ == "__main__":
    main()
