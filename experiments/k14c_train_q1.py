#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-14c: обучение головы q1 на КАНОНИЧЕСКОМ кэше целей. Stage 4 плана K-14.

ЧТО ОБУЧАЕТСЯ. Ровно три группы весов: `depth_rvq_norms.0`,
`depth_rvq_heads.0` и — в вариантах с самообусловливанием — `feedback.0`.
Backbone, Joint12, q0, кодек и уровень q2 заморожены. Состав задаётся
`configure_joint_depth_rvq(stage="q1", variant=...)` и не имеет режима по
умолчанию: молча выбранный набор обучаемых весов — это молча другой
эксперимент.

ЦЕЛИ НЕ ПЕРЕСЧИТЫВАЮТСЯ. Метка q1* — ближайший код к остатку
z_e - E0[q0hat], и у части позиций два кода почти равноудалены (§42, разрыв
5e-08 при величинах 1e-02). Пересчёт в тренере дал бы двум прогонам РАЗНЫЕ
задачи, и разность между ними нельзя было бы приписать порядку данных. Здесь
читается готовый целочисленный кэш K-14b с проверкой отпечатков.

ВАРИАНТЫ (`--variant`):
    main         условные цели из кэша, feedback включён
    static       ИСТИННЫЕ коды q1 как цели, feedback включён — контроль на
                 расхождение меток K-8
    no_feedback  условные цели, feedback ОБХОДИТСЯ в forward, а не просто
                 заморожен

ДВА ПРОГОНА — ЭТО TRAINING/DATA-ORDER SEEDS, А НЕ ДВЕ ИНИЦИАЛИЗАЦИИ. Поздние
головы инициализируются детерминированно: копия `action_lm_head`, нулевая
проекция feedback, копия исходной финальной нормы. Различается только порядок
данных, и называть это независимыми инициализациями неверно.

ОТБОР ЭПОХИ ТОЛЬКО ПО val_sel. Подтверждающая половина открывается ОДИН РАЗ,
после выбора. Gate 4 считается на ней по заранее записанной формуле:

    C = (E_A0 - E_model) / (E_A0 - E_oracle) >= 0.20

где E_A0 и E_oracle берутся ИЗ АРТЕФАКТА ОРАКУЛА, а не из этого прогона.
"""
import argparse
import copy
import hashlib
import json
import os
import sys
import time

import numpy as np

GATE4_MIN_CAPTURE = 0.20


def sha12(path, chunk=1 << 22):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()[:12]


def arr_sha(a):
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


def capture_fraction(e_a0, e_model, e_oracle):
    """Какую долю оракульного выигрыша забрала обученная голова.

    ЗНАМЕНАТЕЛЬ ПРОВЕРЯЕТСЯ ЯВНО. При e_oracle >= e_a0 выигрыша нет вовсе, и
    отношение было бы делением на ноль или отрицательным числом с видом доли.
    Отрицательный результат при исправном знаменателе законен и означает, что
    модель хуже опоры — это надо видеть, а не прятать.
    """
    for nm, v in (("e_a0", e_a0), ("e_model", e_model),
                  ("e_oracle", e_oracle)):
        if v is None or not np.isfinite(float(v)):
            raise SystemExit(f"{nm} = {v}: не число")
    gap = float(e_a0) - float(e_oracle)
    if gap <= 1e-12:
        raise SystemExit(
            f"оракул не лучше опоры: E_A0 {e_a0}, E_oracle {e_oracle}. "
            f"Доля захвата не определена")
    return (float(e_a0) - float(e_model)) / gap


def check_gate4(e_a0, e_model, e_oracle, threshold=GATE4_MIN_CAPTURE):
    """Gate 4 плана K-14. Порог записан ДО обучения и здесь не подбирается."""
    c = capture_fraction(e_a0, e_model, e_oracle)
    limit = float(e_a0) - threshold * (float(e_a0) - float(e_oracle))
    ok = bool(c >= threshold)
    return dict(capture=c, threshold=float(threshold), passed=ok,
                rms_limit=limit, e_a0=float(e_a0), e_model=float(e_model),
                e_oracle=float(e_oracle))


def select_epoch(history):
    """Эпоха выбирается ТОЛЬКО по val_sel. Эпоха 0 — полноценный кандидат.

    Если начальное состояние головы окажется лучше всего обученного, это
    результат, а не повод исключить его из рассмотрения.
    """
    if not history:
        raise SystemExit("пустая история обучения")
    for h in history:
        if h.get("val_sel") is None or not np.isfinite(float(h["val_sel"])):
            raise SystemExit(f"эпоха {h.get('epoch')}: val_sel = "
                             f"{h.get('val_sel')}")
        if "val_confirm" in h:
            raise SystemExit(
                f"эпоха {h.get('epoch')}: в истории есть val_confirm. "
                f"Подтверждающая половина открывается один раз ПОСЛЕ выбора, "
                f"иначе отбор идёт по ней")
    best = min(history, key=lambda h: (float(h["val_sel"]), int(h["epoch"])))
    return int(best["epoch"]), float(best["val_sel"])


def state_sha(named):
    """Отпечаток набора именованных тензоров, в фиксированном порядке."""
    h = hashlib.sha1()
    for k in sorted(named):
        v = named[k]
        h.update(k.encode())
        h.update(np.ascontiguousarray(
            np.asarray(v, dtype=np.float64)).tobytes())
    return h.hexdigest()[:12]


def snapshot(named):
    """Копия обучаемых весов. ИМЕННО КОПИЯ.

    Без клонирования снимок указывал бы на те же тензоры, что продолжают
    меняться, и «восстановление лучшей эпохи» вернуло бы последнюю.
    """
    return {k: v.detach().clone() for k, v in named.items()}


def restore(named, snap):
    """Вернуть веса снимка И ПРОВЕРИТЬ, что вернулись именно они."""
    import torch
    miss = sorted(set(snap) - set(named))
    extra = sorted(set(named) - set(snap))
    if miss or extra:
        raise SystemExit(f"снимок не соответствует модели: нет {miss[:3]}, "
                         f"лишние {extra[:3]}")
    with torch.no_grad():
        for k, v in snap.items():
            named[k].data.copy_(v)
    for k, v in snap.items():
        if not torch.equal(named[k].detach().cpu(), v.detach().cpu()):
            raise SystemExit(f"восстановление неточно по {k}")
    return True


def check_cache_manifest(man, *, oracle_sha1, cache, ckpt, expect_sha1):
    """Кэш целей обязан быть тем, что построен на данных пройденного гейта."""
    # ВСЕ ОТПЕЧАТКИ ОБЯЗАТЕЛЬНЫ. Условие «сверить, если поле есть» означает
    # «согласиться, если отпечатка нет», и это уже четвёртый случай того же
    # шаблона в проекте. Поля перечислены здесь, а не проверяются по месту,
    # чтобы забыть одно было невозможно.
    need = ("kind", "labels_sha1", "q1_sha1", "rows_sha1", "device",
            "oracle_sha1", "cache", "ckpt", "n_rows", "parts",
            "q0hat_sha1", "ktrue_sha1", "split_sha1", "cache_meta_sha1",
            "source_cache_sha1", "keys_sha1", "codebooks_sha1",
            "codec_state_sha1", "decoder_probe",
            # ПРОИСХОЖДЕНИЕ ЧЕРНОВИКА. Цели равны Q1(z_e - E0[q0]), поэтому
            # кэш целей без указания, какой это был q0, не определён.
            "q0_source", "q0_canonical", "plan_sha1", "plan_batch",
            "gate_r_sha1", "q0_manifest_sha1", "q0_npz_sha1", "q0_run_id")
    miss = [k for k in need if man.get(k) is None]
    if miss:
        raise SystemExit(f"в манифесте кэша нет полей {miss}")
    if man.get("q0_canonical") is not True:
        raise SystemExit(
            f"кэш целей построен от черновика {man.get('q0_source')}, а не от "
            f"канонического q0 K-14d. Сентябрьский q0hat сегодня побитово не "
            f"повторяется: обучаться на целях от него значит обучаться на "
            f"мишени, которую нельзя пересчитать")
    if man["kind"] != "canonical_q1_targets":
        raise SystemExit(f"манифест описывает {man['kind']}, а нужен "
                         f"canonical_q1_targets")
    bad = []
    if man["labels_sha1"] != expect_sha1:
        bad.append(f"labels_sha1 {man['labels_sha1']} против фактического "
                   f"{expect_sha1}")
    if man["oracle_sha1"] != oracle_sha1:
        bad.append(f"кэш построен по артефакту гейта {man['oracle_sha1']}, "
                   f"а подан {oracle_sha1}")
    for k, v in (("cache", cache), ("ckpt", ckpt)):
        if str(man[k]) != str(v):
            bad.append(f"{k}: {man[k]} против {v}")
    if bad:
        raise SystemExit("манифест кэша целей не сходится: " + "; ".join(bad))
    return True


def selftest():
    # --- ДОЛЯ ЗАХВАТА И ПОРОГ --------------------------------------------
    assert abs(capture_fraction(1.0, 0.8, 0.0) - 0.2) < 1e-12
    assert abs(capture_fraction(1.0, 1.2, 0.0) + 0.2) < 1e-12   # хуже опоры
    for a_, m_, o_, why in ((1.0, 0.5, 1.0, "не лучше опоры"),
                            (1.0, 0.5, 2.0, "не лучше опоры"),
                            (float("nan"), 0.5, 0.0, "e_a0"),
                            (1.0, float("inf"), 0.0, "e_model")):
        try:
            capture_fraction(a_, m_, o_)
        except SystemExit as e:
            assert why in str(e), (why, str(e))
        else:
            raise AssertionError(f"пропущено: {why}")

    # Значения округлённые, поэтому и предел здесь округлённый. Настоящий
    # порог берётся из артефакта оракула с полной точностью: на его числах
    # (0.1443760…, 0.0826230…) он равен 0.1320250185.
    g = check_gate4(0.14438, 0.13, 0.08262)
    assert g["passed"] and abs(g["rms_limit"] - 0.132028) < 1e-6, g
    assert not check_gate4(0.14438, 0.14, 0.08262)["passed"]
    # ровно на пороге — проходит
    lim = check_gate4(0.14438, 0.0, 0.08262)["rms_limit"]
    assert check_gate4(0.14438, lim, 0.08262)["passed"]

    # --- ОТБОР ЭПОХИ ------------------------------------------------------
    hist = [dict(epoch=0, val_sel=0.20), dict(epoch=1, val_sel=0.15),
            dict(epoch=2, val_sel=0.17)]
    assert select_epoch(hist) == (1, 0.15)
    # эпоха 0 — полноценный кандидат
    assert select_epoch([dict(epoch=0, val_sel=0.1),
                         dict(epoch=1, val_sel=0.2)])[0] == 0
    # при равенстве берётся РАННЯЯ эпоха, а не случайная
    assert select_epoch([dict(epoch=0, val_sel=0.1),
                         dict(epoch=1, val_sel=0.1)])[0] == 0
    for h_, why in (([], "пустая"),
                    ([dict(epoch=0, val_sel=None)], "val_sel"),
                    ([dict(epoch=0, val_sel=0.1, val_confirm=0.2)],
                     "val_confirm")):
        try:
            select_epoch(h_)
        except SystemExit as e:
            assert why in str(e), (why, str(e))
        else:
            raise AssertionError(f"отбор принял: {why}")

    # --- СНИМОК И ВОССТАНОВЛЕНИЕ ЛУЧШЕЙ ЭПОХИ ----------------------------
    # Регрессия на главный блокер: эпоха выбиралась, а веса оставались от
    # последней, и Gate 4 считался не на той модели.
    import torch as _t
    par = {"w": _t.nn.Parameter(_t.zeros(3)),
           "b": _t.nn.Parameter(_t.ones(2))}
    snap0 = snapshot(par)
    sha0 = state_sha({k: v.detach().numpy() for k, v in par.items()})
    with _t.no_grad():
        par["w"] += 5.0
    assert state_sha({k: v.detach().numpy()
                      for k, v in par.items()}) != sha0
    assert float(snap0["w"].abs().max()) == 0.0, "снимок изменился вместе с весами"
    restore(par, snap0)
    assert state_sha({k: v.detach().numpy()
                      for k, v in par.items()}) == sha0
    try:
        restore({"w": par["w"]}, snap0)
    except SystemExit as e:
        assert "не соответствует" in str(e), e
    else:
        raise AssertionError("снимок чужой формы принят")

    # --- МАНИФЕСТ КЭША ----------------------------------------------------
    man = dict(kind="canonical_q1_targets", labels_sha1="L", q1_sha1="Q",
               rows_sha1="R", device="cuda:0", oracle_sha1="O",
               cache="data/c", ckpt="CK", n_rows=10,
               parts={nm: dict(n_rows=1) for nm in
                      ("train", "val_sel", "val_confirm")},
               q0hat_sha1="A", ktrue_sha1="B", split_sha1="S",
               cache_meta_sha1="M", source_cache_sha1="SC", keys_sha1="K",
               codebooks_sha1="CB", codec_state_sha1="CS",
               decoder_probe="DP", q0_source="k14d_plan", q0_canonical=True,
               plan_sha1="P", plan_batch=8, gate_r_sha1="GR",
               q0_manifest_sha1="QM", q0_npz_sha1="QN", q0_run_id="QR")
    mk = dict(oracle_sha1="O", cache="data/c", ckpt="CK", expect_sha1="L")
    check_cache_manifest(man, **mk)
    for patch, why in ((dict(kind="other"), "canonical_q1_targets"),
                       (dict(labels_sha1="Z"), "labels_sha1"),
                       (dict(oracle_sha1="Z"), "артефакту гейта"),
                       (dict(cache="d"), "cache"),
                       # ЦЕЛИ ОТ НЕВОСПРОИЗВОДИМОГО ЧЕРНОВИКА НЕ ПРИНИМАЮТСЯ
                       (dict(q0_canonical=False, q0_source="k11a_legacy"),
                        "канонического q0")):
        try:
            check_cache_manifest(dict(man, **patch), **mk)
        except SystemExit as e:
            assert why in str(e), (why, str(e))
        else:
            raise AssertionError(f"манифест принят при: {why}")
    # ОТСУТСТВИЕ ЛЮБОГО ОБЯЗАТЕЛЬНОГО ПОЛЯ — ОТКАЗ. Перечислять их по одному
    # значит проверить, что список в схеме действительно работает, а не
    # выглядит работающим.
    for key in sorted(man):
        try:
            check_cache_manifest({k: v for k, v in man.items() if k != key},
                                 **mk)
        except SystemExit as e:
            assert "нет полей" in str(e) and key in str(e), (key, str(e))
        else:
            raise AssertionError(f"манифест без {key} принят")
    print("самопроверка k14c_train_q1 пройдена")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--q1-cache", default="data/k14b/q1_cache")
    ap.add_argument("--oracle", default="reports/k14a/oracle_cache_cuda0.json")
    ap.add_argument("--gate-r", default="reports/k14d/gate_r.json",
                    help="доказательство воспроизводимости плана батчей; без "
                         "него цели не определены однозначно")
    ap.add_argument("--cache", default="data/k11a_joint12")
    ap.add_argument("--joint-ckpt", default="data/k9d_ep3.pt")
    ap.add_argument("--ckpt",
                    default="ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO")
    ap.add_argument("--root", default="third_party/actioncodec")
    ap.add_argument("--cfg-path", default="config/eval/bar.yaml")
    # НЕ required: иначе `--selftest` требовал бы вариант и сид, которых у
    # проверки чистых функций нет. Обязательность проверяется ниже, после
    # ветки самопроверки.
    ap.add_argument("--variant", choices=("main", "static", "no_feedback"))
    ap.add_argument("--seed", type=int, default=None,
                    help="training/data-order seed: поздние головы "
                         "инициализируются детерминированно, различается "
                         "только порядок данных")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--lambda-action", type=float, default=1.0)
    ap.add_argument("--lambda-fb", type=float, default=1e-4)
    ap.add_argument("--grip-weight", type=float, default=1.0,
                    help="вес канала схвата в потере действия; фиксируется "
                         "до первого запуска")
    ap.add_argument("--q0", default="data/k14d/q0_b8_e0.npz",
                    help="канонический черновик K-14d: ТОТ ЖЕ массив, от "
                         "которого построены цели и посчитан Gate 2")

    ap.add_argument("--smoke", action="store_true",
                    help="проверка связности: train и val_sel урезаются, "
                         "подтверждающая половина НЕ ЧИТАЕТСЯ вовсе, Gate 4 "
                         "не считается, результат не годится как голова")
    ap.add_argument("--limit", type=int, default=0,
                    help="ограничить smoke ЦЕЛЫМИ каноническими батчами "
                         "(число батчей на часть, только со --smoke). "
                         "Урезание по строкам с последующей перенарезкой "
                         "дало бы неполные батчи, которых нет в плане, а "
                         "значит другой q0")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="разрешить прогон на незакоммиченном коде; "
                         "канонический результат так получать нельзя")
    ap.add_argument("--allow-device-drift", action="store_true",
                    help="разрешить карту, отличную от той, на которой "
                         "построен q0. ТОЛЬКО со --smoke или "
                         "--eval-checkpoint: это диагностика переносимости, "
                         "а не канонический прогон. Побочно она её и меряет: "
                         "сверка q0 внутри прогона побитовая, и если на "
                         "другой карте она проходит, переносимость есть")
    ap.add_argument("--eval-checkpoint", default="",
                    help="измерить сохранённую голову на val_sel (CE, top-1 "
                         "по кодам, RMS) и выйти; подтверждающая половина не "
                         "открывается")
    ap.add_argument("--summary", default="",
                    help="куда записать машинную сводку прогона (json)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    miss = [n for n, v in (("--variant", a.variant), ("--seed", a.seed))
            if v is None]
    if miss:
        raise SystemExit(f"нужны {miss}: вариант и training-сид задаются "
                         f"явно, умолчаний у них нет")
    if a.allow_device_drift and not (a.smoke or a.eval_checkpoint):
        raise SystemExit(
            "--allow-device-drift допустим только со --smoke или "
            "--eval-checkpoint: это диагностика, а не канонический прогон")
    if a.limit and not a.smoke:
        raise SystemExit("--limit допустим только вместе со --smoke: "
                         "укороченный train в каноническом прогоне дал бы "
                         "голову, обученную не на том наборе")
    out_p = a.out or (f"data/k14c/smoke_{a.variant}_s{a.seed}.pt" if a.smoke
                      else f"data/k14c/q1_{a.variant}_s{a.seed}.pt")
    # В РЕЖИМЕ ОЦЕНКИ ГОЛОВА НЕ ПИШЕТСЯ ВОВСЕ, и проверка на существование
    # выходного файла запрещала измерять ровно тот чекпойнт, ради которого
    # режим и заведён: его путь совпадает с тем, который прогон записал бы.
    if os.path.exists(out_p) and not a.eval_checkpoint:
        raise SystemExit(f"{out_p} уже существует: голова не перезаписывается "
                         f"молча")

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(a.root)
    for p in (root, os.path.join(root, "src"), here,
              os.path.abspath("experiments")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import k14_common as kc
    import k12b_protocol as kb

    def code_state():
        """Коммит и отпечатки всего, что влияет на обучение.

        СНИМАЕТСЯ НА СТАРТЕ И ПЕРЕПРОВЕРЯЕТСЯ ПЕРЕД ПОДТВЕРЖДЕНИЕМ. Семь
        часов — достаточный срок, чтобы файл успели поправить; результат,
        полученный наполовину одним кодом и наполовину другим, не относится
        ни к одному из них.
        """
        head, dirty, _arte = kc.check_code_clean(a.allow_dirty)
        return head, kb.code_version([
            os.path.abspath(__file__),
            os.path.join(here, "k14_common.py"),
            os.path.join(here, "depth_rvq_joint12.py"),
            os.path.join(here, "depth_rvq_vla.py"),
            os.path.join(here, "joint12_vla.py"),
            os.path.join(here, "k14b_build_q1_cache.py")]), bool(dirty)

    # FAIL-FAST ДО ЗАГРУЗКИ МОДЕЛИ. Прежде чистота кода только записывалась в
    # чекпойнт полем git_dirty, то есть семичасовой канонический прогон на
    # незакоммиченном коде доводился до конца и принимался.
    git_head0, code_v0, dirty0 = code_state()

    def write_summary(**kw):
        """Машинная сводка прогона. НЕ бинарная: её можно положить в git.

        Чекпойнт весит сотни мегабайт и в репозиторий не кладётся, а без
        какой-либо машинной записи результат существует только в тексте
        отчёта и проверке не поддаётся.
        """
        if not a.summary:
            return
        d_ = dict(kind="k14c_run", variant=a.variant, seed=int(a.seed),
                  smoke=bool(a.smoke), limit=int(a.limit),
                  epochs=int(a.epochs), batch=int(a.batch),
                  git_head=git_head0, git_dirty=bool(dirty0),
                  code_version=code_v0,
                  script_sha1=sha12(os.path.abspath(__file__)), **kw)
        os.makedirs(os.path.dirname(os.path.abspath(a.summary)) or ".",
                    exist_ok=True)
        t_ = a.summary + f".tmp.{os.getpid()}"
        json.dump(d_, open(t_, "w"), ensure_ascii=False, indent=1,
                  default=str)
        os.replace(t_, a.summary)
        print(f"  сводка: {a.summary}")
    print(f"  код: коммит {git_head0}, "
          f"{len(code_v0)} файлов в версии"
          + ("  (--allow-dirty)" if dirty0 else ""))
    import torch
    import torch.nn.functional as F
    import k11a_build_hicora_cache as k11a
    import k11b_hicora_identity as k11b
    from depth_rvq_joint12 import (make_joint_depth_rvq_class,
                                   code_contribution)
    from depth_rvq_vla import straight_through
    from joint12_vla import make_joint12_class
    import actioncodec  # noqa: F401
    from smolvla.bar import SmolVLABlockwiseAR
    import inspect
    from utils import (ACTION_Q01, ACTION_Q99, STATE_Q01, STATE_Q99,
                       VisionLanguageActionProcessor, dict_apply, get_cfg,
                       prompt_template)

    # ОТПЕЧАТОК bar.py БЕРЁТСЯ ОТ ФАКТИЧЕСКИ ИМПОРТИРОВАННОГО КЛАССА.
    # Собранный вручную путь был неверен (`src/smolvla/bar.py` вместо
    # `smolvla/bar.py`), и поле молча становилось None — то есть провенанс
    # сегментированного прохода отсутствовал, а выглядел записанным.
    bar_p = inspect.getfile(SmolVLABlockwiseAR)
    if not bar_p or not os.path.exists(bar_p):
        raise SystemExit(f"не удалось определить файл SmolVLABlockwiseAR "
                         f"({bar_p}): без его отпечатка провенанс прохода "
                         f"неполон")
    bar_sha = sha12(bar_p)

    dev = torch.device(a.device)
    dt = getattr(torch, a.dtype)
    # СИД ЗАПУСКА НЕ ДОЛЖЕН ДОХОДИТЬ ДО ИНИЦИАЛИЗАЦИИ. Поздние головы и так
    # собираются детерминированно, но глобальный `manual_seed(a.seed)`
    # оставлял канал, по которому два прогона могли бы разойтись не только
    # порядком данных. Глобальные генераторы фиксируются постоянным значением,
    # а сид запуска применяется РОВНО в одном месте — к порядку батчей.
    INIT_SEED = 0
    torch.manual_seed(INIT_SEED)
    np.random.seed(INIT_SEED)
    max_act_q = np.maximum(np.abs(ACTION_Q01), np.abs(ACTION_Q99))
    H_EXEC = 8

    # --- кэш целей и его привязка -------------------------------------------
    npz_p = a.q1_cache + ".npz"
    man_p = a.q1_cache + ".manifest.json"
    man = json.load(open(man_p))
    check_cache_manifest(man, oracle_sha1=sha12(a.oracle), cache=a.cache,
                         ckpt=a.ckpt, expect_sha1=sha12(npz_p))
    with np.load(npz_p, allow_pickle=True) as z:
        if sorted(z.files) != ["part", "q1", "rows"]:
            raise SystemExit(f"в кэше целей массивы {sorted(z.files)}")
        q1_raw, rows_raw, part_all = z["q1"], z["rows"], z["part"].astype(str)
    # ПРОВЕРКИ ДО ПРИВЕДЕНИЯ ТИПА. `np.asarray(x, np.int64)` молча примет и
    # числа с плавающей точкой, и отрицательные, и чужие формы — а дальше это
    # будет выглядеть исправными метками.
    if q1_raw.dtype.kind != "i" or rows_raw.dtype.kind != "i":
        raise SystemExit(f"типы массивов: q1 {q1_raw.dtype}, rows "
                         f"{rows_raw.dtype}; оба обязаны быть целыми")
    if arr_sha(q1_raw) != man["q1_sha1"]:
        raise SystemExit(f"массив целей имеет sha {arr_sha(q1_raw)}, в "
                         f"манифесте {man['q1_sha1']}")
    if arr_sha(rows_raw.astype(np.int64)) != man["rows_sha1"]:
        raise SystemExit("массив строк не совпал с отпечатком манифеста")
    if q1_raw.ndim != 2 or q1_raw.shape[0] != rows_raw.shape[0] \
            or q1_raw.shape[0] != len(part_all):
        raise SystemExit(f"формы: q1 {q1_raw.shape}, rows {rows_raw.shape}, "
                         f"part {part_all.shape}")
    if int(man["n_rows"]) != int(q1_raw.shape[0]):
        raise SystemExit(f"строк {q1_raw.shape[0]}, в манифесте "
                         f"{man['n_rows']}")
    rows_all = np.asarray(rows_raw, np.int64)
    q1_all = np.asarray(q1_raw, np.int64)
    if len(np.unique(rows_all)) != len(rows_all):
        raise SystemExit("номера строк в кэше целей повторяются")
    if rows_all.min() < 0:
        raise SystemExit(f"отрицательный номер строки {rows_all.min()}")
    if set(np.unique(part_all)) != {"train", "val_sel", "val_confirm"}:
        raise SystemExit(f"части кэша целей: {sorted(set(part_all))}")
    for nm_ in ("train", "val_sel", "val_confirm"):
        # ЧИСЛО СТРОК В МАНИФЕСТЕ ОБЯЗАТЕЛЬНО. Прежнее «сверить, если поле
        # есть» означало «согласиться, если поля нет»: манифест без разбивки
        # по частям проходил бы молча, и обучение шло бы на другом составе.
        pm_ = ((man.get("parts") or {}).get(nm_) or {})
        if pm_.get("n_rows") is None:
            raise SystemExit(f"в манифесте кэша нет числа строк части {nm_}")
        want_n, got_n = int(pm_["n_rows"]), int((part_all == nm_).sum())
        if got_n != want_n:
            raise SystemExit(f"часть {nm_}: {got_n} строк, в манифесте "
                             f"{want_n}")
    # АРТЕФАКТ GATE R ПЕРЕПРОВЕРЯЕТСЯ ЗДЕСЬ, а не принимается по отметке в
    # манифесте кэша: отметка говорит лишь, что при сборке кэша он был.
    gr_info = kc.check_gate_r(a.gate_r, expect_plan_sha1=man["plan_sha1"])
    if gr_info["gate_r_sha1"] != man["gate_r_sha1"]:
        raise SystemExit(f"Gate R {gr_info['gate_r_sha1']}, кэш целей "
                         f"построен при {man['gate_r_sha1']}")
    # РАЗМЕР МИКРОБАТЧА ОБЯЗАН СОВПАСТЬ С ТЕМ, ПРИ КОТОРОМ ПОСТРОЕН q0.
    # Промпты дополняются слева до самого длинного в батче, поэтому состав
    # батча входит в вычисление. При другом размере тренер считал бы свой q0,
    # не равный кэшу, и сверка с A0 сравнивала бы разные величины.
    if int(a.batch) != int(gr_info["batch"]):
        raise SystemExit(
            f"--batch {a.batch}, а q0 построен при {gr_info['batch']}. Размер "
            f"прямого микробатча — часть определения задачи (решение 5.1), а "
            f"не настройка скорости: копите градиент, но считайте по "
            f"{gr_info['batch']}")
    print(f"  Gate R: {a.gate_r}, план {gr_info['plan_sha1']}, порядки "
          f"{gr_info['exec_order_seeds']}, микробатч {gr_info['batch']}")

    orc = json.load(open(a.oracle))
    # ВХОДНЫЕ МАССИВЫ СВЕРЯЮТСЯ С ТЕМИ, НА КОТОРЫХ ПОСТРОЕН КЭШ ЦЕЛЕЙ.
    # Отпечатка одного .npz мало: `ktrue` определяет цели варианта `static`,
    # `q0hat` — остаток, от которого цели считались, `codebooks` — и цели, и
    # потерю действия. Подмена любого из них меняет эксперимент молча.
    for nm, key in (("q0hat", "q0hat_sha1"), ("ktrue", "ktrue_sha1"),
                    ("split", "split_sha1")):
        got_ = k11a.file_sha1(f"{a.cache}.{nm}.npy")
        if got_ != man[key]:
            raise SystemExit(f"{nm}.npy имеет sha {got_}, кэш целей построен "
                             f"на {man[key]}")
    mm_ = k11a.file_sha1(f"{a.cache}.meta.json")
    if mm_ != man["cache_meta_sha1"]:
        raise SystemExit(f"meta.json кэша имеет sha {mm_}, цели построены на "
                         f"{man['cache_meta_sha1']}")
    print(f"  цели: {npz_p}, sha {man['labels_sha1']}, построены в режиме "
          f"{man['device']} по гейту {man['oracle_sha1']}")

    # --- исходные данные ----------------------------------------------------
    meta = json.load(open(f"{a.cache}.meta.json"))
    src = meta["cache"]
    d = np.load(src, allow_pickle=True)
    cmeta = json.loads(str(d["meta"]))
    N = int(meta["n_obs"])
    epi, stp = np.asarray(d["episode"])[:N], np.asarray(d["step"])[:N]
    keys_sha = hashlib.sha1(np.ascontiguousarray(
        np.stack([epi, stp])).tobytes()).hexdigest()[:12]
    if keys_sha != meta.get("keys_sha1"):
        raise SystemExit(f"ключи наблюдений {keys_sha} против "
                         f"{meta.get('keys_sha1')}")
    src_sha = sha12(src)
    if src_sha != man["source_cache_sha1"]:
        raise SystemExit(f"исходный кэш K-9a {src_sha}, цели построены на "
                         f"{man['source_cache_sha1']}: массив действий "
                         f"определяет и цели, и потерю действия")
    if keys_sha != man["keys_sha1"]:
        raise SystemExit(f"ключи наблюдений {keys_sha}, цели построены на "
                         f"{man['keys_sha1']}")
    ACT = np.asarray(d["action"])[:N]
    offs = np.asarray(d["pos_offset"])[:N].astype(np.int64)
    tsk = np.asarray(d["task"])[:N]
    ktrue = np.load(f"{a.cache}.ktrue.npy", mmap_mode="r")
    E = np.load(f"{a.cache}.codebooks.npy")

    # --- КАНОНИЧЕСКИЙ ЧЕРНОВИК И СОХРАНЁННЫЙ ПЛАН ---------------------------
    # Сентябрьский q0hat K-11a здесь больше не участвует ни в сверке, ни в
    # аудите: он побитово не воспроизводится, и расхождение с ним измерено
    # (0.17-0.24% позиций). Тренер стоит на том же массиве, от которого
    # построены цели и посчитан Gate 2.
    q0_can, q0_defined, q0_man, q0_prov = kc.load_canonical_q0(
        a.q0, gate_r_path=a.gate_r, n_obs=N, keys_sha=keys_sha,
        cache_meta_sha1=k11a.file_sha1(f"{a.cache}.meta.json"))
    bad_q0 = [k_ for k_ in ("q0_manifest_sha1", "q0_npz_sha1", "q0_run_id",
                            "plan_sha1", "plan_batch", "gate_r_sha1")
              if str(man.get(k_)) != str(q0_prov.get(k_))]
    if bad_q0:
        raise SystemExit(
            f"кэш целей построен на другом черновике: расходятся {bad_q0}. "
            f"Цели равны Q1(z_e - E0[q0]); от другого q0 это другие цели")
    plan_all = kc.load_plan(a.q0, q0_man)
    # КАРТА И РЕЖИМ ВЫЧИСЛЕНИЙ. Побитового совпадения q0 недостаточно, чтобы
    # считать задачу той же: q0 — это argmax, он грубее скрытых состояний.
    # Два устройства могут дать одинаковые коды и при этом слегка разные h12,
    # из которых обучается q1. Пока переносимость не измерена, канонический
    # прогон обязан идти на той же карте и в том же режиме.
    rt_now = dict(device=str(dev), gpu_uuid=kc.gpu_uuid(dev, torch),
                  compute_dtype=a.dtype, torch_version=str(torch.__version__),
                  cuda_version=str(getattr(torch.version, "cuda", None)),
                  tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),
                  tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),
                  cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
                  cudnn_benchmark=bool(torch.backends.cudnn.benchmark))
    drift = [k_ for k_, v_ in rt_now.items()
             if str(q0_man.get(k_)) != str(v_)]
    if drift:
        msg = ("прогон идёт в другом режиме, чем построен q0: "
               + "; ".join(f"{k_}: сейчас {rt_now[k_]}, у q0 "
                           f"{q0_man.get(k_)}" for k_ in drift))
        if not (a.allow_device_drift and (a.smoke or a.eval_checkpoint)):
            raise SystemExit(
                msg + ". Для диагностики переносимости: --smoke вместе с "
                      "--allow-device-drift; канонический прогон так вести "
                      "нельзя")
        print(f"  ДИАГНОСТИКА ПЕРЕНОСИМОСТИ: {msg}")
    # СМЕЩЕНИЕ, ЗАПИСАННОЕ В ПЛАНЕ, ОБЯЗАНО СОВПАСТЬ СО СМЕЩЕНИЕМ ДАННЫХ.
    # `build_inputs` принимает одно position_offset на батч; если план говорит
    # одно, а кэш — другое, вход читается не с той позиции, и расхождение
    # выглядело бы как ошибка модели.
    for nm_, po_, sel_ in plan_all:
        if not bool((offs[sel_] == po_).all()):
            raise SystemExit(
                f"батч части {nm_} заявлен со смещением {po_}, а строки "
                f"{sel_[:3]} имеют {sorted(set(int(x) for x in offs[sel_]))}")
    if int(a.batch) != int(q0_man["batch"]):
        raise SystemExit(f"--batch {a.batch}, план построен при "
                         f"{q0_man['batch']}")
    print(f"  черновик: {q0_prov['q0_npz']} ({q0_prov['q0_npz_sha1']}), "
          f"план {q0_prov['plan_sha1']}, {len(plan_all)} батчей, "
          f"Gate R {q0_prov['gate_r_sha1']}")

    img_p = os.path.join(os.path.dirname(src), cmeta["images_file"])
    IMG = np.load(img_p, mmap_mode="r")
    if IMG.shape[0] < N or IMG.dtype != np.uint8:
        raise SystemExit(f"кадры {IMG.shape} {IMG.dtype}: не те")
    # СОСТОЯНИЯ — через общий строгий загрузчик. Он один для K-14a, K-14b,
    # K-14c и K-14d: три предыдущих раза одна и та же проверка писалась
    # заново и каждый раз оказывалась fail-open по какому-нибудь полю.
    # ПРОИСХОЖДЕНИЕ ДАННЫХ — из meta кэша K-11a через вычитыватель K-11b:
    # в meta исходного npz этих полей нет по построению, там лежит только путь
    # к манифесту разбиения.
    ds_repo, ds_rev = k11b.dataset_source(meta)
    st_n, sm, st_shas = kc.load_states(src, N, ds_repo, ds_rev, keys_sha,
                                       STATE_Q01, STATE_Q99)
    print(f"  данные: {N} наблюдений, кадры {IMG.shape[1:]}, состояния "
          f"{st_n.shape[1]}-мерные (ключи {sm.get('keys_sha1')}, "
          f"sha {st_shas['state_npy']})")

    # --- модель -------------------------------------------------------------
    cfg = get_cfg(os.path.join(root, a.cfg_path))
    # ПУТЬ К БАЗОВОЙ МОДЕЛИ ПЕРЕОПРЕДЕЛЯЕТСЯ, КАК В K-11a И K-12d. В конфиге
    # авторов зашит абсолютный путь с их машины; без подмены `from_pretrained`
    # пытается трактовать его как имя репозитория на HuggingFace.
    cfg.TRAINING.ckpt_dir = a.ckpt
    cfg.MODEL.vlm.kwargs.pretrained_model_name_or_path = a.ckpt
    Base = make_joint12_class(SmolVLABlockwiseAR)
    model = Base.from_pretrained(**cfg.MODEL.vlm.kwargs).to(dev, dt).eval()
    proc = VisionLanguageActionProcessor.from_pretrained(
        a.ckpt, trust_remote_code=True, mode="discrete")
    model.init_joint_fast(depth=int(meta["depth"]), head_dtype=torch.float32)
    # ИСХОДНАЯ ФИНАЛЬНАЯ НОРМА СНИМАЕТСЯ ДО ЗАГРУЗКИ Joint12: он перезаписывает
    # action_expert.norm, обученную читать h12, а поздние выходы должны
    # стартовать с нормы, калиброванной на полной глубине.
    refine_norm = copy.deepcopy(model.action_expert.norm)
    j_sha = k11a.file_sha1(a.joint_ckpt)
    if j_sha != (meta.get("source") or {}).get("weights_sha1"):
        raise SystemExit(f"Joint12 {j_sha}, кэш собран на "
                         f"{(meta.get('source') or {}).get('weights_sha1')}")
    obj = torch.load(a.joint_ckpt, map_location="cpu", weights_only=False)
    state = obj["state"]
    own = dict(model.named_parameters())
    stray = [k for k in state if not any(k.startswith(p) or k == p.rstrip(".")
                                         for p in model.trainable_prefixes)]
    missing = [k for k in own if own[k].requires_grad and k not in state]
    if stray or missing:
        raise SystemExit(f"Joint12 загружается не строго: лишние {stray[:3]}, "
                         f"нет {missing[:3]}")
    with torch.no_grad():
        for k, v in state.items():
            own[k].data = v.to(dev, torch.float32)
    print(f"  Joint12 загружен строго: {len(state)} тензоров, sha {j_sha}")

    model.__class__ = make_joint_depth_rvq_class(type(model))
    model.init_joint_depth_rvq(refine_norm=refine_norm,
                               books=torch.from_numpy(E),
                               feedback=(a.variant != "no_feedback"),
                               verbose_init=False)
    info = model.configure_joint_depth_rvq(stage="q1", variant=a.variant)
    codec = proc.action_processor
    codec = (codec if hasattr(codec, "vq") else codec.codec).to(dev).eval()
    for p_ in codec.parameters():
        p_.requires_grad_(False)
    # КОДЕК СВЕРЯЕТСЯ ПО ВЕСАМ И ПОВЕДЕНИЮ. Имя чекпойнта на HuggingFace не
    # гарантирует содержимого: изменившийся артефакт под тем же именем
    # сдвинул бы потерю действия и Gate 4.
    with torch.no_grad():
        ii = torch.arange(int(codec.vocab_size), device=dev).unsqueeze(0)
        Ecur = torch.stack([q.out_project(q.decode_code(ii))[0]
                            for q in codec.vq.quantizers]).float()
    if float((Ecur.cpu() - torch.from_numpy(E)).abs().max()) > 1e-5:
        raise SystemExit("книги кодека разошлись с кэшем")
    cs_now = k11a.state_sha1(codec)
    dp_now = k11a.decoder_probe(codec, Ecur.to(dev), dev)
    cb_now = arr_sha(np.asarray(E, np.float32))
    for nm, cur, want in (("codebooks_sha1", cb_now, man["codebooks_sha1"]),
                          ("codec_state_sha1", cs_now,
                           man["codec_state_sha1"]),
                          ("decoder_probe", dp_now, man["decoder_probe"])):
        if cur != want:
            raise SystemExit(f"{nm}: сейчас {cur}, цели построены при {want}")
    print(f"  кодек сверен: книги {cb_now}, веса {cs_now}, проба {dp_now}")

    # ПРОДОЛЖЕНИЯ НЕТ НАМЕРЕННО. Оно требовало бы переносить состояние Adam,
    # порядок данных, историю и факт уже открытой подтверждающей половины;
    # без этого `--resume` был бы тёплым стартом под видом продолжения, а
    # `val_confirm` открывалась бы повторно. Прогон идёт целиком или не идёт.

    # --- батчи --------------------------------------------------------------
    keep = ("train", "val_sel") if a.smoke else ("train", "val_sel",
                                                 "val_confirm")
    # БАТЧИ БЕРУТСЯ ИЗ ПЛАНА, А НЕ НАРЕЗАЮТСЯ ЗАНОВО. Состав батча входит в
    # вычисление (дополнение слева до самого длинного промпта), поэтому
    # собственная нарезка совпадала бы с канонической лишь случайно — и
    # заведомо расходилась бы на укороченном smoke.
    batches = {nm: [(po, sel) for n_, po, sel in plan_all if n_ == nm]
               for nm in keep}
    empty = [nm for nm in keep if not batches[nm]]
    if empty:
        raise SystemExit(f"в плане нет батчей частей {empty}")
    sets = {nm: rows_all[part_all == nm] for nm in keep}
    # ЦЕЛИ РАСКЛАДЫВАЮТСЯ ПО НОМЕРАМ СТРОК КЭША K-11a: батчи формируются по
    # смещению позиций, а не по порядку в кэше целей, и брать цель по позиции
    # в массиве было бы сопоставлением не тех строк.
    q1_of_pos = np.full((N, q1_all.shape[1]), -1, np.int64)
    q1_of_pos[rows_all] = q1_all
    if a.limit:
        # ЦЕЛЫЕ КАНОНИЧЕСКИЕ БАТЧИ. Урезание по строкам с перенарезкой давало
        # бы неполный хвостовой батч по каждому смещению — батч, которого в
        # плане нет, а значит другой вход и другой q0.
        for nm in batches:
            batches[nm] = batches[nm][:a.limit]
    for nm in keep:
        rows_nm = np.concatenate([sel for _po, sel in batches[nm]])
        if not a.limit and not np.array_equal(np.sort(rows_nm),
                                              np.sort(sets[nm])):
            raise SystemExit(
                f"часть {nm}: строки плана не совпадают со строками кэша "
                f"целей — цели и черновик построены на разных наборах")
        sets[nm] = rows_nm
    print("  части: " + ", ".join(
        f"{k} {len(sets[k])} строк в {len(batches[k])} батчах" for k in keep))

    def build(po, sel):
        image = torch.from_numpy(np.asarray(IMG[sel]))
        msgs = []
        for gi in sel:
            m = prompt_template(
                st_n[gi], None, str(tsk[gi]),
                mode=cfg.MODEL.vla_processor.kwargs.mode,
                action_vocab_size=cfg.MODEL.action_processor.vocab_size,
                action_token_len=cfg.MODEL.action_processor.token_len)
            m[1]["content"] = m[1]["content"][1:]
            msgs.append(m)
        texts = proc.apply_chat_template(msgs, add_generation_prompt=True)
        b = proc(text=texts, images=[[image[k].numpy()]
                                     for k in range(len(sel))],
                 return_tensors="pt", padding=True, padding_side="left",
                 action_processor_kwargs={"embodiment_ids": 0})
        return dict_apply(lambda x: x.to(dev, dt), b)

    books = model.depth_rvq_books
    ac16 = torch.autocast(device_type=dev.type, dtype=dt)
    q0_bad, q0_tot, q0_margins, q0_margins_ok = [0], [0], [], []

    def targets(sel):
        if a.variant == "static":
            # КОНТРОЛЬ K-8: истинные коды q1 без учёта ошибки префикса.
            return torch.from_numpy(
                np.asarray(ktrue[sel])[:, 1, :].astype(np.int64)).to(dev)
        t = q1_of_pos[sel]
        if (t < 0).any():
            raise SystemExit("строка вне канонического кэша целей")
        return torch.from_numpy(t).to(dev)

    def decode_actions(z):
        x, _ = codec._decode(z.float(), embodiment_ids=0)
        return x[..., :7].float()

    def run_batch(po, sel, train, no_q1=False, mix_ps=None, mix_rng=None):
        """`no_q1` — ОПОРА A0: действие декодируется из одного E0[q0].

        Без неё нельзя сказать, помогает ли голова ВООБЩЕ. Числа оракула
        сняты на других строках (train там подвыбирался), и сравнивать с ними
        срез значит сравнивать разные наборы. Опора должна считаться на тех
        же строках тем же кодом.
        """
        b = build(po, sel)
        with ac16:
            v_, p_ = model.build_inputs(position_offset=po, **b)
            out = model.forward_joint_depth_rvq(
                vlm_inputs_embeds=v_, attention_mask=b.get("attention_mask"),
                position_ids=p_, mode="medium")
        lg = out["logits"][1].float()
        q0 = out["pred_codes"][0]
        # Q0 ОБЯЗАН СОВПАСТЬ С КАНОНИЧЕСКИМ: цели построены как
        # Q1(z_e - E0[q0]) от артефакта K-14d, и если модель выдаёт другой
        # черновик, они относятся к другому остатку. Сравнение идёт с
        # заверенным Gate R массивом, а НЕ с сентябрьским q0hat K-11a:
        # последний сегодня побитово не воспроизводится, и сверка с ним
        # заведомо расходилась бы на измеренных 0.17-0.24% позиций.
        cq0 = torch.from_numpy(
            np.asarray(q0_can[sel]).astype(np.int64)).to(dev)
        d_q0 = int((q0 != cq0).sum())
        # ЗНАМЕНАТЕЛЬ СЧИТАЕТСЯ ВСЕГДА. Прежде он увеличивался внутри ветки
        # расхождения, то есть был числом позиций в «плохих» батчах, и доля
        # завышалась во столько раз, во сколько батчей без расхождений больше.
        q0_tot[0] += int(q0.numel())
        if d_q0:
            # ЗАПАС МЕЖДУ ПЕРВЫМ И ВТОРЫМ ЛОГИТОМ В РАСХОДЯЩИХСЯ ПОЗИЦИЯХ.
            # Он отличает грань от настоящего расхождения путей: при запасе
            # порядка единицы последнего разряда речь о почти равных
            # кандидатах, при большом — о разных вычислениях.
            with torch.no_grad():
                l0 = out["logits"][0].float()
                two = torch.topk(l0, 2, dim=-1).values
                marg = (two[..., 0] - two[..., 1])
                m = (q0 != cq0)
                q0_margins.extend(
                    [float(x) for x in marg[m].detach().cpu().numpy()])
                q0_margins_ok.append(float(marg[~m].median())
                                     if int((~m).sum()) else float("nan"))
            q0_bad[0] += d_q0
            frac = q0_bad[0] / max(q0_tot[0], 1)
            # ДОПУСКА НЕТ. Он был временной мерой, пока канонический q0 не
            # существовал и сравнение шло с невоспроизводимым кэшем. Теперь
            # черновик заверен Gate R, и любое расхождение означает, что
            # исполняется не тот план или не та арифметика, а не «почти то же».
            raise SystemExit(
                f"q0 модели разошёлся с каноническим: {q0_bad[0]} из "
                f"{q0_tot[0]} позиций ({100 * frac:.4f}%). Запас логитов в "
                f"этих позициях: {sorted(q0_margins)[:5]}. Цели построены от "
                f"{q0_prov['q0_npz']} (план {q0_prov['plan_sha1']}, "
                f"Gate R {q0_prov['gate_r_sha1']}) и относятся к другому "
                f"остатку")
        tg = targets(sel)
        ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tg.reshape(-1))
        # E0 СТРОИТСЯ ОТ ФАКТИЧЕСКОГО q0 МОДЕЛИ. Они только что сверены с
        # кэшем, так что значения те же, но брать надо то, что модель выдала:
        # иначе оценивался бы гибрид, которого модель не производила.
        e0 = books[0][q0.long()].float()
        if train:
            emb, _, _ = straight_through(lg, books[1].float(), tau=1.0)
        else:
            emb = books[1][lg.argmax(-1)].float()
        a_hat = decode_actions(e0 if no_q1 else e0 + emb)
        a_true = torch.from_numpy(
            np.asarray(ACT[sel], np.float32)).to(dev)[..., :7]
        # ВЕСА ОБУЧЕНИЯ И ВЕСА ГЕЙТА — РАЗНЫЕ, И ЭТО ИЗМЕРЯЕТСЯ.
        # В потере все каналы равны (кроме схвата), в метрике Gate 4 они
        # взвешены физическим масштабом max_act_q. При равных сырых ошибках
        # вращательные каналы получают в потере в 6-21 раз больший вес, чем
        # в метрике, по которой судят. Раньше в истории лежала только сумма,
        # и разойтись этим двум величинам было нечем помешать и нечем
        # заметить.
        w = torch.ones(7, device=dev, dtype=torch.float32)
        w[6] = float(a.grip_weight)
        raw = a_hat[:, :H_EXEC] - a_true[:, :H_EXEC]
        dd = raw * w
        act_loss = (dd ** 2).mean()
        with torch.no_grad():
            wg = torch.as_tensor(max_act_q[:7], device=dev,
                                 dtype=torch.float32).clone()
            wg[-1] = 1.0
            ch_sq = ((raw * wg) ** 2).sum(dim=(0, 1))     # по каналам, гейт
            ch_sq_raw = (raw ** 2).sum(dim=(0, 1))        # по каналам, сырые
            n_el = int(raw.shape[0] * raw.shape[1])
            stat = dict(n_tok=int(lg.shape[0] * lg.shape[1]),
                        correct=float((lg.argmax(-1) == tg).sum()),
                        ce_sum=float(ce) * int(lg.shape[0] * lg.shape[1]),
                        act_train=float(act_loss) * int(dd.numel()),
                        act_gate=float(ch_sq.sum()), n_el=n_el,
                        ch_gate=ch_sq.detach().cpu().numpy(),
                        ch_raw=ch_sq_raw.detach().cpu().numpy())
        fb_reg = torch.zeros((), device=dev)
        if a.variant != "no_feedback":
            for p_ in model.depth_rvq_feedback[0].parameters():
                fb_reg = fb_reg + (p_.float() ** 2).sum()
        loss = ce + a.lambda_action * act_loss + a.lambda_fb * fb_reg
        if not torch.isfinite(loss):
            raise SystemExit(
                f"потеря не число: CE {float(ce)}, действие "
                f"{float(act_loss)}, регуляризатор {float(fb_reg)}")
        if mix_ps is not None:
            # СКОЛЬКО ТОЧНОСТИ НУЖНО. Доля `p` предсказанных кодов
            # заменяется оракульной меткой; p=0 — сама голова, p=1 — оракул.
            # Кривая RMS(p) отвечает, достижим ли порог в принципе и при
            # какой точности, а не «выучится ли голова ещё немного».
            # Всё считается ОДНИМ прямым проходом: дорог только он, декодер
            # дёшев.
            with torch.no_grad():
                pred = lg.argmax(-1)
                mix = {}
                for pp in mix_ps:
                    m_ = torch.from_numpy(
                        mix_rng.random(tuple(pred.shape)) < float(pp)).to(dev)
                    q1m = torch.where(m_, tg, pred)
                    am = decode_actions(e0 + books[1][q1m].float())
                    dm = (am[:, :H_EXEC] - a_true[:, :H_EXEC]) * wg
                    mix[float(pp)] = (float((dm ** 2).sum()), int(dm.numel()))
                stat["mix"] = mix
        return loss, ce, act_loss, a_hat.detach(), a_true, d_q0, stat

    ev_extra = {}

    def evaluate(bs, no_q1=False):
        """RMS первых восьми действий в единицах робота, argmax без ST.

        ПОБОЧНО СОБИРАЕТ CE И TOP-1 ПО КОДАМ. Печаталась только суммарная
        обучающая функция ce + L_action + L_fb, и по ней нельзя сказать,
        улучшается ли предсказание кодов: три слагаемых могли двигаться
        по-разному. Утверждение «коды точнее, действие нет» требует
        раздельного измерения, иначе это домысел.
        """
        se, n = 0.0, 0
        # СУММЫ, А НЕ СРЕДНЕЕ СРЕДНИХ. Хвостовой батч короче полного, и
        # равный вес дал бы ему завышенное влияние — знакомая проекту ошибка.
        acc = dict(ce_sum=0.0, correct=0.0, n_tok=0, act_train=0.0,
                   act_gate=0.0, n_el=0, rows=0, batches=0)
        ch_g, ch_r = np.zeros(7), np.zeros(7)
        model.eval()
        with torch.no_grad():
            for po, sel in bs:
                _l, _c, _al, a_hat, a_true, _d, st_ = run_batch(
                    po, sel, False, no_q1=no_q1)
                for k_ in ("ce_sum", "correct", "n_tok", "act_train",
                           "act_gate", "n_el"):
                    acc[k_] += st_[k_]
                ch_g += st_["ch_gate"]; ch_r += st_["ch_raw"]
                acc["rows"] += int(len(sel)); acc["batches"] += 1
                q = torch.as_tensor(max_act_q[:7], device=dev,
                                    dtype=torch.float32).clone()
                q[-1] = 1.0
                dd = (a_hat[:, :H_EXEC] - a_true[:, :H_EXEC]) * q
                se += float((dd ** 2).sum())
                n += dd.numel()
        nt = max(acc["n_tok"], 1)          # токенов кода q1
        ne = max(acc["n_el"], 1)           # строк x H_EXEC
        nel7 = ne * 7                      # то же по всем семи каналам
        ev_extra.clear()
        ev_extra.update(
            ce=acc["ce_sum"] / nt, top1=acc["correct"] / nt,
            n_tok=acc["n_tok"], rows=acc["rows"], batches=acc["batches"],
            # СКВ в весах ОБУЧЕНИЯ и в весах ГЕЙТА — на одних и тех же
            # строках. Если они расходятся, оптимизатор честно снижал свою
            # величину, не двигаясь в сторону той, по которой судят.
            rms_train_w=float(np.sqrt(acc["act_train"] / nel7)),
            rms_gate_w=float(np.sqrt(acc["act_gate"] / nel7)),
            ch_rms_gate=[float(x) for x in np.sqrt(ch_g / ne)],
            ch_rms_raw=[float(x) for x in np.sqrt(ch_r / ne)])
        return float(np.sqrt(se / max(n, 1)))

    if a.eval_checkpoint:
        # Снимок НАЧАЛЬНОГО состояния тех же весов — опора сравнения: без неё
        # непонятно, что именно дало обучение, а что было с самого начала.
        snap0_eval = snapshot({n_: p_ for n_, p_ in model.named_parameters()
                               if n_ in set(info["names"])})
        # ТОЛЬКО ИЗМЕРЕНИЕ, БЕЗ ОБУЧЕНИЯ И БЕЗ ПОДТВЕРЖДАЮЩЕЙ ПОЛОВИНЫ.
        # Отвечает на один вопрос: расходятся ли точность по кодам и ошибка
        # действия. Подтверждение здесь не открывается ни при каких условиях:
        # оно одноразовое и тратится только в каноническом прогоне.
        # ЗАГРУЗКА СТРОГАЯ. Прежде сверялись только неизвестные ключи и
        # формы: пустой или неполный state прошёл бы молча, и измерение
        # относилось бы к голове, которую никто не обучал. Множество ключей
        # обязано совпасть ТОЧНО с белым списком этапа, а отпечаток весов
        # после загрузки — с записанным в чекпойнте.
        ck_ = torch.load(a.eval_checkpoint, map_location="cpu",
                         weights_only=False)
        # Поля перечислены по тому, что канонический прогон ДЕЙСТВИТЕЛЬНО
        # пишет: у него `stage`, а не `kind`. Требовать несуществующее поле
        # — это отказ на ровном месте, а не строгость.
        need_ck = ("stage", "variant", "seed", "state", "trainable_names",
                   "selected_state_sha1", "q0_prov")
        miss_f = [k for k in need_ck if ck_.get(k) is None]
        if miss_f:
            raise SystemExit(f"в чекпойнте нет полей {miss_f}")
        if str(ck_["stage"]) != "q1":
            raise SystemExit(f"чекпойнт этапа {ck_['stage']}, ожидался q1")
        if str(ck_["variant"]) != str(a.variant):
            raise SystemExit(f"чекпойнт варианта {ck_['variant']}, запрошен "
                             f"{a.variant}: это другая голова")
        st_ = ck_["state"]
        want_ = set(info["names"])
        if set(ck_["trainable_names"]) != want_:
            raise SystemExit(
                f"белый список чекпойнта не совпадает с белым списком этапа: "
                f"лишние {sorted(set(ck_['trainable_names']) - want_)[:5]}, "
                f"нет {sorted(want_ - set(ck_['trainable_names']))[:5]}")
        if set(st_) != want_:
            raise SystemExit(
                f"ключи чекпойнта не совпадают с белым списком этапа: "
                f"лишние {sorted(set(st_) - want_)[:5]}, "
                f"нет {sorted(want_ - set(st_))[:5]}")
        own_ = dict(model.named_parameters())
        with torch.no_grad():
            for k_, v_ in st_.items():
                if tuple(own_[k_].shape) != tuple(v_.shape):
                    raise SystemExit(f"форма {k_}: {tuple(v_.shape)} против "
                                     f"{tuple(own_[k_].shape)}")
                if not torch.isfinite(v_).all():
                    raise SystemExit(f"в {k_} есть nan или inf")
                own_[k_].data.copy_(v_.to(own_[k_].device, own_[k_].dtype))
        got_sha = state_sha({k_: own_[k_].detach().float().cpu().numpy()
                             for k_ in want_})
        if got_sha != ck_["selected_state_sha1"]:
            raise SystemExit(
                f"после загрузки веса имеют отпечаток {got_sha}, в чекпойнте "
                f"{ck_['selected_state_sha1']}: загрузилось не то состояние")
        # ПРОИСХОЖДЕНИЕ ЧЕРНОВИКА У ЧЕКПОЙНТА И У ТЕКУЩЕГО ПРОГОНА — ОДНО.
        bad_p = [k_ for k_ in ("q0_manifest_sha1", "q0_npz_sha1", "plan_sha1",
                               "gate_r_sha1")
                 if str((ck_["q0_prov"] or {}).get(k_))
                 != str(q0_prov.get(k_))]
        if bad_p:
            raise SystemExit(f"чекпойнт обучен на другом черновике: {bad_p}")
        print(f"  загружен чекпойнт {a.eval_checkpoint}: {len(st_)} тензоров, "
              f"вариант {ck_.get('variant')}, сид {ck_.get('seed')}, "
              f"эпоха {ck_.get('selected_epoch')}, отпечаток весов "
              f"{got_sha} совпал")
        # СРЕЗ TRAIN РАВНОГО РАЗМЕРА. Обучающая и валидационная метрики
        # должны быть ОДНОЙ величиной на наборах одного размера, иначе
        # «обучающая падает, валидационная стоит» нечем проверить. Именно
        # этот признак отличает нехватку данных (train сильно лучше val) от
        # недоученности (обе плохи и примерно равны).
        # СРЕЗ TRAIN БЕРЁТСЯ РАВНОМЕРНО ПО ВСЕМУ ПЛАНУ, А НЕ С НАЧАЛА.
        # План упорядочен по смещению позиций, поэтому первые N батчей — это
        # строки одного-двух смещений, то есть другой состав задач, а не
        # случайная часть train. Равномерная выборка индексов детерминирована
        # и покрывает все смещения пропорционально их доле.
        n_v = len(batches["val_sel"])
        tr_all = batches["train"]
        take = np.unique(np.linspace(0, len(tr_all) - 1,
                                     min(n_v, len(tr_all))).astype(int))
        ev_sets = {"val_sel": batches["val_sel"],
                   "train (срез)": [tr_all[i] for i in take]}
        print(f"    срез train: {len(take)} батчей из {len(tr_all)}, "
              f"смещений {len(set(int(tr_all[i][0]) for i in take))} из "
              f"{len(set(int(b[0]) for b in tr_all))}")
        a0_ = (((orc.get("parts") or {}).get("val_sel") or {})
               .get("vs_action.A0") or {}).get("rms")
        or_ = (((orc.get("parts") or {}).get("val_sel") or {})
               .get("vs_action.A01_ze") or {}).get("rms")
        curves = {}

        def mix_sweep(tag):
            """RMS на val_sel при подмене доли кодов оракульными.

            СЧИТАЕТСЯ ДЛЯ КАЖДОГО СОСТОЯНИЯ ВЕСОВ ОТДЕЛЬНО. Первая версия
            стояла после цикла, который последним восстанавливает начальные
            веса, и потому мерила НЕОБУЧЕННУЮ голову. Поймала это встроенная
            сверка «p=0 обязан совпасть с обученной головой» — поэтому она и
            печатается рядом с кривой, а не проверяется мысленно.

            Смысл величины: при доле p позиций код берётся оракульный, на
            остальных — тот, что выдала голова. Кривая отвечает, какая
            точность нужна для порога, и насколько дорого обходится ошибка.
            """
            ps_ = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
            rng_m = np.random.default_rng(0)
            acc_m = {float(x): [0.0, 0] for x in ps_}
            with torch.no_grad():
                for po, sel in batches["val_sel"]:
                    _l, _c, _a, _ah, _at, _d, st_m = run_batch(
                        po, sel, False, mix_ps=ps_, mix_rng=rng_m)
                    for k_, (sq_, n_) in st_m["mix"].items():
                        acc_m[k_][0] += sq_; acc_m[k_][1] += n_
            print(f"\n  RMS-8 на val_sel, коды {tag} с подменой доли "
                  f"оракульными:")
            cur = {}
            for k_ in ps_:
                sq_, n_ = acc_m[float(k_)]
                r_ = float(np.sqrt(sq_ / max(n_, 1)))
                c_ = ((a0_ - r_) / (a0_ - or_) if (a0_ and or_)
                      else float("nan"))
                cur[float(k_)] = dict(rms=r_, capture=c_)
                mark = "  <- порог" if c_ >= 0.20 else ""
                print(f"    доля оракула {100 * k_:5.1f}%   RMS {r_:.6f}   "
                      f"C = {c_:+.4f}{mark}")
            curves[tag] = cur
            return cur

        rows_ = {}
        # ОПОРА A0 НА ТЕХ ЖЕ СТРОКАХ. Считается один раз: она не зависит от
        # весов головы — q1 просто не применяется.
        for nm_, bs_ in ev_sets.items():
            r_ = evaluate(bs_, no_q1=True)
            rows_[f"опора A0 / {nm_}"] = dict(rms=r_, n_batches=len(bs_),
                                              **dict(ev_extra))
            print(f"    {'опора A0':12s} {nm_:14s} RMS-8 {r_:.6f}  "
                  f"({ev_extra['rows']} строк)")
            print("                 по каналам (веса гейта): "
                  + " ".join(f"{x:.4f}" for x in ev_extra["ch_rms_gate"]))
        for tag, st0 in (("обученная", True), ("до обучения", False)):
            if not st0:
                with torch.no_grad():
                    for k_, v_ in snap0_eval.items():
                        own_[k_].data.copy_(v_.to(own_[k_].device,
                                                  own_[k_].dtype))
            for nm_, bs_ in ev_sets.items():
                r_ = evaluate(bs_)
                rows_[f"{tag} / {nm_}"] = dict(rms=r_, n_batches=len(bs_),
                                               **dict(ev_extra))
                print(f"    {tag:12s} {nm_:14s} RMS-8 {r_:.6f}  "
                      f"CE {ev_extra['ce']:.5f}  "
                      f"top-1 {100 * ev_extra['top1']:.2f}%  "
                      f"({ev_extra['rows']} строк, {ev_extra['n_tok']} "
                      f"токенов)")
                print(f"                 {'':14s} СКВ в весах обучения "
                      f"{ev_extra['rms_train_w']:.6f}, в весах гейта "
                      f"{ev_extra['rms_gate_w']:.6f}")
                print("                 по каналам (веса гейта): "
                      + " ".join(f"{x:.4f}"
                                 for x in ev_extra["ch_rms_gate"]))
            mix_sweep(tag)
        if a0_ and or_:
            for tag, d_ in rows_.items():
                if tag.endswith("val_sel"):
                    print(f"    {tag:28s} захват C = "
                          f"{(a0_ - d_['rms']) / (a0_ - or_):+.4f}")
        # ПОМОГАЕТ ЛИ ГОЛОВА ВООБЩЕ — на каждом наборе отдельно.
        for nm_ in ev_sets:
            b_ = rows_.get(f"опора A0 / {nm_}")
            h_ = rows_.get(f"обученная / {nm_}")
            if b_ and h_:
                d_ = b_["rms"] - h_["rms"]
                print(f"    {nm_:14s} голова против опоры A0: "
                      f"{h_['rms']:.6f} против {b_['rms']:.6f} "
                      f"({'лучше' if d_ > 0 else 'ХУЖЕ'} на {abs(d_):.6f})")
        tr_, vl_ = rows_.get("обученная / train (срез)"), \
            rows_.get("обученная / val_sel")
        if tr_ and vl_:
            print(f"\n  РАЗРЫВ ОБУЧЕНИЕ/ВАЛИДАЦИЯ: CE {tr_['ce']:.5f} против "
                  f"{vl_['ce']:.5f}, top-1 {100 * tr_['top1']:.2f}% против "
                  f"{100 * vl_['top1']:.2f}%")
            print("  Большой разрыв -> упёрлись в данные. Малый разрыв при "
                  "низком top-1 на обоих -> упёрлись в бюджет или ёмкость.")
        write_summary(outcome="eval_only", eval_val_sel=rows_,
                      accuracy_curves=curves,
                      checkpoint=a.eval_checkpoint, runtime=rt_now,
                      q0_prov=q0_prov, e_a0_val_sel=a0_,
                      e_oracle_val_sel=or_)
        return 0

    named_tr = {n_: p_ for n_, p_ in model.named_parameters()
                if n_ in set(info["names"])}
    params = [named_tr[n_] for n_ in info["names"]]
    init_sha = state_sha({k: v.detach().float().cpu().numpy()
                          for k, v in named_tr.items()})
    print(f"  начальное состояние обучаемых весов: sha {init_sha}. Оно "
          f"детерминировано и НЕ зависит от --seed: сид задаёт только порядок "
          f"данных")
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    rng = np.random.default_rng(a.seed)
    hist = []
    t0 = time.time()
    e0_val = evaluate(batches["val_sel"])
    hist.append(dict(epoch=0, train=None, val_sel=e0_val))
    # СНИМОК ЭПОХИ 0 — ПОЛНОЦЕННЫЙ КАНДИДАТ. Если начальное состояние головы
    # окажется лучшим, оно и будет восстановлено.
    best = dict(epoch=0, val_sel=e0_val, state=snapshot(named_tr))
    print(f"  эпоха 0 (без обучения): val_sel RMS-8 {e0_val:.6f}")
    for ep in range(1, int(a.epochs) + 1):
        order = list(batches["train"])
        t_ep = time.time()
        rng.shuffle(order)
        run, nb, dq = 0.0, 0, 0
        for po, sel in order:
            opt.zero_grad(set_to_none=True)
            loss, ce, al, _ah, _at, d_q0, _st = run_batch(po, sel, True)
            loss.backward()
            # ГРАДИЕНТ ОБЯЗАН ДОЙТИ ДО КАЖДОГО РАЗРЕШЁННОГО ВЕСА И БЫТЬ
            # КОНЕЧНЫМ. Отсутствующий градиент означает, что часть головы не
            # участвует в потере, и обучается не то, что заявлено.
            if nb == 0:
                nog = [n_ for n_, p_ in zip(info["names"], params)
                       if p_.grad is None]
                nf = [n_ for n_, p_ in zip(info["names"], params)
                      if p_.grad is not None
                      and not torch.isfinite(p_.grad).all()]
                if nog or nf:
                    raise SystemExit(f"градиенты: нет у {nog[:3]}, "
                                     f"нечисловые у {nf[:3]}")
                if ep == 1:
                    # ПОЛОЖИТЕЛЬНАЯ СТРОКА, А НЕ ТОЛЬКО ОТСУТСТВИЕ ОТКАЗА.
                    # По логу должно быть видно, что проверка исполнялась.
                    print(f"    первый шаг: q0 совпал с кэшем, потеря "
                          f"конечна, градиент есть у всех "
                          f"{len(params)} обучаемых тензоров")
            opt.step()
            run += float(loss.detach()); nb += 1; dq += d_q0
            # ПРОГРЕСС ВНУТРИ ЭПОХИ, С ЯВНЫМ СБРОСОМ БУФЕРА. Эпоха идёт часы;
            # без этого лог молчит, и работающий прогон неотличим от
            # зависшего. flush обязателен: при перенаправлении в файл stdout
            # буферизуется блоками, и строки эпох (десятки байт) не дошли бы
            # до файла до самого выхода процесса.
            if nb % 250 == 0:
                el = (time.time() - t_ep) / 60
                print(f"    эпоха {ep}: батч {nb}/{len(order)}, потеря "
                      f"{run / nb:.5f}, {el:.1f} мин, осталось "
                      f"{el * (len(order) - nb) / max(nb, 1):.0f} мин",
                      flush=True)
        v = evaluate(batches["val_sel"])
        hist.append(dict(epoch=ep, train=run / max(nb, 1), val_sel=v))
        mark = ""
        if v < best["val_sel"]:          # СТРОГОЕ улучшение, иначе ранняя
            best = dict(epoch=ep, val_sel=v, state=snapshot(named_tr))
            mark = "  <- лучшая"
        print(f"  эпоха {ep}: потеря {run / max(nb, 1):.5f}, val_sel RMS-8 "
              f"{v:.6f} ({(time.time() - t0) / 60:.1f} мин){mark}")

    best_ep, best_val = select_epoch(hist)
    if best_ep != best["epoch"] or abs(best_val - best["val_sel"]) > 1e-12:
        raise SystemExit(
            f"отбор дал эпоху {best_ep} ({best_val}), а снимок хранит "
            f"{best['epoch']} ({best['val_sel']}): выбор и сохранённые веса "
            f"разошлись")
    # ВЕСА ВЫБРАННОЙ ЭПОХИ ВОССТАНАВЛИВАЮТСЯ ДО ВСЯКОЙ ОЦЕНКИ. Прежде
    # подтверждающая половина считалась на весах ПОСЛЕДНЕЙ эпохи, а в отчёт
    # шёл номер выбранной — Gate 4 относился бы не к той модели.
    restore(named_tr, best["state"])
    re_val = evaluate(batches["val_sel"])
    if abs(re_val - best_val) > 1e-9:
        raise SystemExit(
            f"после восстановления val_sel {re_val:.8f} против {best_val:.8f}: "
            f"восстановлены не те веса")
    sel_sha = state_sha({k: v.detach().float().cpu().numpy()
                         for k, v in named_tr.items()})
    print(f"\n  выбрана эпоха {best_ep} по val_sel ({best_val:.6f}); веса "
          f"восстановлены и сверены, sha {sel_sha}")

    if a.smoke:
        # СТРОКИ ПОДТВЕРЖДАЮЩЕЙ ПОЛОВИНЫ НЕ ОБРАЗУЮТ НАБОРА И НЕ ПРОХОДЯТ
        # ЧЕРЕЗ МОДЕЛЬ; метрика по ним не вычисляется. Сам файл целей и
        # артефакт гейта, разумеется, читаются целиком — но ни одно
        # наблюдение этой половины моделью не обработано.
        print("  РЕЖИМ SMOKE: строки подтверждающей половины через модель не "
              "проходили, метрика по ним не считалась, Gate 4 не вычислялся; "
              "эта голова для эксперимента непригодна")
        write_summary(outcome="smoke", history=hist, runtime=rt_now,
                      q0_prov=q0_prov, q0_mismatch=int(q0_bad[0]),
                      q0_positions=int(q0_tot[0]),
                      initial_trainable_state_sha1=init_sha,
                      selected_state_sha1=sel_sha, selected_epoch=best_ep,
                      val_sel=best_val,
                      parts={k_: dict(rows=int(len(sets[k_])),
                                      batches=len(batches[k_]))
                             for k_ in sets})
        if a.out:
            torch.save(dict(kind="smoke", stage="q1", variant=a.variant,
                            seed=int(a.seed), history=hist,
                            initial_trainable_state_sha1=init_sha,
                            q0_mismatch=int(q0_bad[0]),
                            q0_positions=int(q0_tot[0]), q0_prov=q0_prov,
                            selected_state_sha1=sel_sha,
                            note="проверка связности; как источник весов "
                                 "для канонического прогона непригодна"),
                       a.out)
            print(f"  сохранено: {a.out}")
        return 0

    # ПЕРЕД ОТКРЫТИЕМ ПОДТВЕРЖДАЮЩЕЙ ПОЛОВИНЫ КОД СВЕРЯЕТСЯ ЗАНОВО. Она
    # открывается один раз, и открывать её результатом, полученным частично
    # другим кодом, значит потратить её впустую.
    git_head1, code_v1, _d1 = code_state()
    if git_head1 != git_head0 or code_v1 != code_v0:
        # ВЕСА СОХРАНЯЮТСЯ, ПОДТВЕРЖДЕНИЕ НЕ ОТКРЫВАЕТСЯ. Отказ обязан стоить
        # ровно того, что он защищает. Защищается одноразовая подтверждающая
        # половина, а не семь часов обучения: выбрасывать обученную голову
        # из-за того, что рядом сменился коммит, значит наказывать за
        # постороннее. Голова помечается непригодной для гейта и сохраняется.
        changed = [k for k in code_v0 if code_v0[k] != code_v1.get(k)]
        print(f"  КОД ИЗМЕНИЛСЯ ВО ВРЕМЯ ОБУЧЕНИЯ: коммит {git_head0} -> "
              f"{git_head1}, файлы {changed or 'те же'}. Подтверждающая "
              f"половина НЕ открывается: результат не относится ни к одной "
              f"из версий целиком. Веса сохраняются, для Gate 4 они "
              f"непригодны")
        tmp = out_p + f".tmp.{os.getpid()}"
        torch.save(dict(kind="q1_head_unconfirmed", stage="q1",
                        variant=a.variant, seed=int(a.seed),
                        state={k: v.detach().cpu()
                               for k, v in model.state_dict().items()
                               if k in set(info["names"])},
                        history=hist, selected_epoch=best_ep,
                        val_sel=best_val,
                        initial_trainable_state_sha1=init_sha,
                        selected_state_sha1=sel_sha, q0_prov=q0_prov,
                        git_head_start=git_head0, git_head_end=git_head1,
                        code_version_start=code_v0, code_version_end=code_v1,
                        runtime=rt_now,
                        note="подтверждающая половина не открывалась: код "
                             "изменился во время обучения"), tmp)
        os.replace(tmp, out_p)
        print(f"  сохранено: {out_p}")
        write_summary(outcome="code_changed_during_run", history=hist,
                      runtime=rt_now, q0_prov=q0_prov, val_sel=best_val,
                      selected_epoch=best_ep, selected_state_sha1=sel_sha,
                      git_head_end=git_head1, code_version_end=code_v1,
                      checkpoint=out_p)
        # КОД 5 — НЕ 4 И НЕ 0: прогон не завершён по протоколу, и бегунок
        # обязан остановить цепочку, а не считать это отрицательным
        # результатом.
        return 5

    e_conf = evaluate(batches["val_confirm"])
    po_ = (orc.get("parts") or {}).get("val_confirm") or {}
    e_a0 = (po_.get("vs_action.A0") or {}).get("rms")
    e_or = (po_.get("vs_action.A01_ze") or {}).get("rms")
    g4 = check_gate4(e_a0, e_conf, e_or)
    print(f"  ПОДТВЕРЖДЕНИЕ (открыто один раз): RMS-8 {e_conf:.6f}")
    print(f"  Gate 4: захват {g4['capture']:.4f} при пороге "
          f"{g4['threshold']}, предел RMS {g4['rms_limit']:.6f} -> "
          f"{'ПРОЙДЕН' if g4['passed'] else 'НЕ ПРОЙДЕН'}")

    os.makedirs(os.path.dirname(os.path.abspath(out_p)) or ".", exist_ok=True)
    ck = dict(
        kind="q1_head",
        state={k: v.detach().cpu() for k, v in model.state_dict().items()
               if k in set(info["names"])},
        stage="q1", variant=a.variant, seed=int(a.seed),
        seed_kind="training/data-order: поздние головы инициализируются "
                  "детерминированно",
        trainable_names=info["names"], n_tensors=info["n_tensors"],
        n_params=info["n_params"], feedback_mask=info["feedback_mask"],
        q1_cache=a.q1_cache, q1_cache_sha1=man["labels_sha1"],
        oracle=a.oracle, oracle_sha1=sha12(a.oracle),
        joint_ckpt=a.joint_ckpt, joint_sha1=j_sha, cache=a.cache, ckpt=a.ckpt,
        q1_manifest_sha1=sha12(man_p), source_cache_sha1=src_sha,
        q0hat_sha1=k11a.file_sha1(f"{a.cache}.q0hat.npy"),
        ktrue_sha1=k11a.file_sha1(f"{a.cache}.ktrue.npy"),
        codebooks_sha1=cb_now,
        codec_state_sha1=cs_now, decoder_probe=dp_now,
        initial_trainable_state_sha1=init_sha,
        q0_mismatch=int(q0_bad[0]), q0_positions=int(q0_tot[0]),
        q0_prov=q0_prov,
        selected_state_sha1=sel_sha,
        selected_epoch=best_ep, val_sel=best_val, val_confirm=e_conf,
        gate4=g4, history=hist, epochs_run=int(a.epochs),
        lr=a.lr, wd=a.wd, batch=int(a.batch),
        lambda_action=a.lambda_action, lambda_fb=a.lambda_fb,
        grip_weight=a.grip_weight, device=str(dev), dtype=a.dtype,
        bar_path=bar_p,
        git_dirty=bool(dirty0),
        # В ВЕРСИЮ КОДА ВХОДИТ ВСЁ, ЧТО ВЛИЯЕТ НА ОБУЧЕНИЕ. Прежде сюда не
        # попадали joint12_vla.py (ранний выход и его норма), depth_rvq_vla.py
        # (straight_through и CodeFeedback) и bar.py (сегментированный проход),
        # хотя изменение любого из них меняет обученную голову.
        code_version=code_v0, git_head=git_head0, runtime=rt_now,
        bar_sha1=bar_sha,
        script_sha1=sha12(os.path.abspath(__file__)))
    tmp = out_p + f".tmp.{os.getpid()}"
    torch.save(ck, tmp)
    os.replace(tmp, out_p)
    print(f"  сохранено: {out_p}")
    write_summary(outcome=("gate4_passed" if g4["passed"]
                           else "gate4_not_passed"),
                  history=hist, runtime=rt_now, q0_prov=q0_prov,
                  q0_mismatch=int(q0_bad[0]), q0_positions=int(q0_tot[0]),
                  initial_trainable_state_sha1=init_sha,
                  selected_state_sha1=sel_sha, selected_epoch=best_ep,
                  val_sel=best_val, val_confirm=e_conf, gate4=g4,
                  e_a0=e_a0, e_oracle=e_or, checkpoint=out_p,
                  q1_cache=a.q1_cache, oracle=a.oracle, gate_r=a.gate_r,
                  parts={k_: dict(rows=int(len(sets[k_])),
                                  batches=len(batches[k_])) for k_ in sets})
    # КОД 4 — ЭТО ЗАВЕРШЁННЫЙ ПРОГОН С НЕПРОЙДЕННЫМ GATE 4, А НЕ ОТКАЗ.
    # Отличать обязательно: иначе бегунок, увидев отрицательный, но валидный
    # результат первой реплики, не запустит вторую — и зарегистрированный
    # дизайн с двумя порядками данных превратится в остановку после
    # просмотра результата.
    return 0 if g4["passed"] else 4


if __name__ == "__main__":
    sys.exit(main())
