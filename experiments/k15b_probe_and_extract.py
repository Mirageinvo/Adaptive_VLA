#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15b, шаг 0: выбрать книгу C1 и цель читателя по ОДНОМУ правилу.

ОДИН ПРОХОД НА СОСТОЯНИЕ, ВСЕ ЧИСЛА НА ОДНИХ СТРОКАХ. Доли и разности
сравнимы только при общей области, поэтому латентная цель, rank-path-цель,
опора декодера и диапазон действий считаются в одном цикле по одним батчам.

ЧТО ИЗМЕРЯЕТСЯ ДЛЯ КАЖДОЙ УНИКАЛЬНОЙ КНИГИ

    a1_tok    z0 + C1[argmin ||r - C1[k]||]      латентная цель
    a1_rank   z0 + C1[k_j*], j* — лучший НОМЕР РАНГА для всей строки
    a1_pol    то, что модель исполняет сейчас
    a1_soft   z0 + E_P[C1]  (декод среднего эмбеддинга)

и для каждого ПУТИ-ЦЕЛИ — RMS, опора декодера и диапазон действий.

ЧТО ТАКОЕ rank-path И ЧЕМ ОН НЕ ЯВЛЯЕТСЯ. Траектория j берёт j-й по
близости код НА КАЖДОЙ позиции чанка, затем по строке выбирается лучший
НОМЕР РАНГА. Это НЕ «лучший код среди top-k»: смеси вида «ранг 1 на первой
позиции, ранг 3 на второй» не рассматриваются, их k^T. Декодер глобален по
чанку, позиции не разделяются, перебор смесей неосуществим. Поэтому поле
зовётся `action_best_rankpath`, а `not_a_full_oracle` говорит про обе
непроверенные области: коды ВНЕ соседства и смеси ВНУТРИ него.

ПОЧЕМУ ВЫБОР КНИГИ ЗАВИСИТ ОТ ЦЕЛИ. Правило «минимум a1_tok среди
прошедших опору» неверно, если целью объявлен rank-path: проверять надо
опору и диапазон ИМЕННО того пути, по которому будет обучаться читатель, и
минимизировать его же RMS. Поэтому сначала для каждой книги определяется
её фактическая цель, потом гейты пути этой цели, и только среди допустимых
берётся минимум.

КНИГА ЗАПИСЫВАЕТСЯ ТОЛЬКО ПОСЛЕ ВСЕХ ГЕЙТОВ. Прежняя версия писала файл до
проверки опоры, и после отказа на диске оставался внешне нормальный
`k15b_c1_selected`, который следующий тренер мог загрузить. Теперь при
отсутствии допустимой книги пишется только отчёт, а в самом файле книги
стоит буквальное `accepted=True`.

ЧЕКПОЙНТ ПРОВЕРЯЕТСЯ FAIL-CLOSED. Записать `source_checkpoint_sha1` в выход
недостаточно: это опознаёт чужой чекпойнт, но не доказывает, что он собран
в той же обстановке. Сверяются одиннадцать полей, включая ПОБИТОВЫЙ
отпечаток замороженного, и отсутствие обязательного поля — отказ.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

# ПРАВИЛО ВЫБОРА ЦЕЛИ, ЗАФИКСИРОВАННОЕ ДО ДАННЫХ. Границы включаются
# СНИЗУ: ровно 5 % уходит в ветку rank-path, и текст это говорит.
TARGET_RULE = (
    (0.02, "latent", "выигрыш < 2 %: латентная цель"),
    (0.05, "latent", "выигрыш в [2 %, 5 %): латентная цель в первом K-15b"),
    (float("inf"), "action_best_rankpath",
     "выигрыш >= 5 %: лучший rank-path внутри латентного соседства, "
     "если полный кэш укладывается в бюджет"),
)
CACHE_BUDGET_HOURS = 6.0      # дороже — остаёмся на латентной цели
ACTION_RANGE_FACTOR = 1.5     # p99 кандидата против p99 набора, поканально
ACTION_CLIP_BOUND = 1.5       # и абсолютный предел нормированной шкалы


def target_decision(relative_gain, projected_cache_hours):
    """Решение по правилу выше. Чистая функция.

    `relative_gain` = (RMS латентной цели - RMS rank-path) / RMS латентной.
    """
    if not np.isfinite(relative_gain):
        raise ValueError(f"выигрыш не число: {relative_gain}")
    for threshold, choice, why in TARGET_RULE:
        if relative_gain < threshold:
            break
    if (choice == "action_best_rankpath"
            and projected_cache_hours > CACHE_BUDGET_HOURS):
        return dict(target="latent_plus_action_term", rule_branch=why,
                    reason=(f"полный кэш оценён в {projected_cache_hours:.1f} ч "
                            f"при бюджете {CACHE_BUDGET_HOURS:.1f} ч, поэтому "
                            f"остаёмся на латентной цели и добавляем малый "
                            f"action-член"),
                    relative_gain=float(relative_gain),
                    projected_cache_hours=float(projected_cache_hours))
    return dict(target=choice, rule_branch=why, reason=why,
                relative_gain=float(relative_gain),
                projected_cache_hours=float(projected_cache_hours))


def target_path_name(target):
    """Какой путь будет учительским при данном решении о цели."""
    return {"latent": "a1_tok",
            "latent_plus_action_term": "a1_tok",
            "action_best_rankpath": "a1_rank"}[target]


def strided(n_total, n_take):
    """Разреженные индексы, либо ВСЕ при n_take <= 0.

    Первые N батчей плана — это первые эпизоды, а не выборка из части.
    """
    if n_total <= 0:
        return []
    if n_take <= 0 or n_take >= n_total:
        return list(range(int(n_total)))
    return sorted(set(int(x) for x in
                      np.linspace(0, n_total - 1, int(n_take)).astype(int)))


def rank_candidates(distances, chosen, k, torch):
    """k РАЗЛИЧНЫХ кодов на позицию: сначала исполняемый, потом ближайшие.

    Прежняя версия подставляла `chosen` в столбец 0 и теряла прежний top-1,
    если `chosen` уже стоял в столбцах 1..k-1: кандидатов оставалось k-1, а
    один дублировался. Здесь `chosen` исключается из остатка, поэтому кодов
    ровно k и все различны.
    """
    vocab = int(distances.shape[-1])
    if k < 1 or k > vocab:
        raise ValueError(f"k={k} вне [1, {vocab}]")
    if k == 1:
        return chosen.unsqueeze(-1)
    near = (-distances).topk(min(k + 1, vocab), dim=-1).indices
    keep = near != chosen.unsqueeze(-1)
    # устойчивая сортировка отправляет исключённый код в конец
    order = torch.argsort((~keep).to(torch.int8), dim=-1, stable=True)
    rest = near.gather(-1, order)[..., :k - 1]
    return torch.cat([chosen.unsqueeze(-1), rest], dim=-1)


def fraction_of_gap(base, target, value):
    """(base - value) / (base - target) или None при неположительном разрыве.

    Когда путь-учитель ХУЖЕ черновика, разрыв отрицателен и доля теряет
    смысл: в финальном прогоне K-15 она выдала +280 % и +202 %.
    """
    gap = float(base) - float(target)
    if gap <= 1e-12:
        return None
    return float((float(base) - float(value)) / gap)


def range_ok(p99_candidate, p99_dataset, absmax_candidate):
    """Два условия, как в K-15: поканальный p99 И абсолютный предел шкалы."""
    by_p99 = all(float(c) <= ACTION_RANGE_FACTOR * float(d)
                 for c, d in zip(p99_candidate, p99_dataset))
    by_clip = all(float(m) <= ACTION_CLIP_BOUND for m in absmax_candidate)
    return dict(passed=bool(by_p99 and by_clip), by_p99=bool(by_p99),
                by_clip=bool(by_clip), factor=ACTION_RANGE_FACTOR,
                clip_bound=ACTION_CLIP_BOUND)


CHECKPOINT_FIELDS = ("joint_sha1", "codec", "code_version", "variant",
                     "batch", "compute_dtype", "q1_init_sha1",
                     "decoder_context", "init_gate_sha1", "q0_prov",
                     "frozen_content_sha")


def check_checkpoint(obj, ctx, a, file_sha1, frozen_sha):
    """Совместимость чекпойнта K-15 с ТЕКУЩИМ окружением. Fail-closed."""
    problems = []
    missing = [f for f in CHECKPOINT_FIELDS if f not in obj]
    if missing:
        problems.append(f"в чекпойнте нет обязательных полей {missing}")

    def same(name, got, want):
        if got != want:
            problems.append(f"{name}: в чекпойнте {got!r}, сейчас {want!r}")

    same("joint_sha1", obj.get("joint_sha1"), ctx.joint_sha)
    same("codec", obj.get("codec"), ctx.codec_fp)
    same("code_version", obj.get("code_version"), ctx.code_version)
    same("variant", str(obj.get("variant")), str(a.variant))
    same("batch", int(obj.get("batch", -1)), int(a.batch))
    same("compute_dtype", str(obj.get("compute_dtype")), str(a.dtype))
    same("q1_init_sha1", obj.get("q1_init_sha1"), file_sha1(a.q1_init))
    same("decoder_context", obj.get("decoder_context"), ctx.decoder_context)
    same("init_gate_sha1", obj.get("init_gate_sha1"),
         ctx.gate_info.get("init_gate_sha1"))
    for key in ("plan_sha1", "q0_manifest_sha1"):
        same(f"q0_prov.{key}", (obj.get("q0_prov") or {}).get(key),
             ctx.q0_prov.get(key))
    # САМАЯ СИЛЬНАЯ ПРОВЕРКА: отпечаток покрывает backbone, q0, C0 и
    # декодер побитово, а не их имена.
    same("frozen_content_sha", obj.get("frozen_content_sha"), frozen_sha)
    return problems


def selftest():
    import torch

    # --- ПРАВИЛО ВЫБОРА ЦЕЛИ ---------------------------------------------
    assert target_decision(0.005, 1.0)["target"] == "latent"
    assert target_decision(0.03, 1.0)["target"] == "latent"
    assert target_decision(0.12, 1.0)["target"] == "action_best_rankpath"
    # ГРАНИЦЫ ВКЛЮЧАЮТСЯ СНИЗУ, и текст правила это говорит
    assert target_decision(0.02, 1.0)["target"] == "latent"
    assert target_decision(0.05, 1.0)["target"] == "action_best_rankpath"
    assert ">= 5 %" in TARGET_RULE[-1][2]
    d = target_decision(0.12, CACHE_BUDGET_HOURS + 0.1)
    assert d["target"] == "latent_plus_action_term" and "бюджете" in d["reason"]
    for bad in (float("nan"), float("inf")):
        try:
            target_decision(bad, 1.0)
        except ValueError as e:
            assert "не число" in str(e), e
        else:
            raise AssertionError(f"принят выигрыш {bad}")
    assert target_path_name("latent") == "a1_tok"
    assert target_path_name("latent_plus_action_term") == "a1_tok"
    assert target_path_name("action_best_rankpath") == "a1_rank"

    # --- ВЫБОР БАТЧЕЙ ----------------------------------------------------
    assert strided(0, 5) == []
    assert strided(7, 0) == list(range(7))        # 0 означает «все»
    assert strided(7, 99) == list(range(7))
    s = strided(100, 5)
    assert s[0] == 0 and s[-1] == 99 and len(s) == 5 and s != list(range(5))

    # --- k РАЗЛИЧНЫХ КАНДИДАТОВ ------------------------------------------
    # Исполняемый код стоит В СЕРЕДИНЕ top-k: прежняя версия потеряла бы
    # здесь один кандидат и продублировала другой.
    d1 = torch.tensor([[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]])
    chosen = torch.tensor([[2]])
    got = rank_candidates(d1, chosen, 4, torch)
    assert got.shape == (1, 1, 4), got.shape
    row = got[0, 0].tolist()
    assert row[0] == 2 and len(set(row)) == 4 and set(row) == {2, 0, 1, 3}, row
    # исполняемый код уже первый — поведение не меняется
    assert rank_candidates(d1, torch.tensor([[0]]), 3,
                           torch)[0, 0].tolist() == [0, 1, 2]
    assert rank_candidates(d1, chosen, 1, torch)[0, 0].tolist() == [2]
    for bad in (0, 7):
        try:
            rank_candidates(d1, chosen, bad, torch)
        except ValueError as e:
            assert "вне" in str(e), e
        else:
            raise AssertionError(f"принято k={bad}")

    # --- ДОЛЯ РАЗРЫВА ----------------------------------------------------
    assert abs(fraction_of_gap(1.0, 0.5, 0.75) - 0.5) < 1e-12
    assert fraction_of_gap(1.0, 1.0, 0.9) is None        # нулевой разрыв
    assert fraction_of_gap(0.105, 0.115, 0.12) is None   # учитель хуже
    assert fraction_of_gap(1.0, 0.0, 1.5) == -0.5

    # --- ДИАПАЗОН: ДВА УСЛОВИЯ, А НЕ ОДНО --------------------------------
    ok = range_ok([0.5, 0.5], [1.0, 1.0], [0.9, 0.9])
    assert ok["passed"] and ok["by_p99"] and ok["by_clip"]
    bad_p99 = range_ok([2.0, 0.5], [1.0, 1.0], [0.9, 0.9])
    assert not bad_p99["passed"] and not bad_p99["by_p99"]
    # p99 в норме, а максимум вылетел за нормированную шкалу
    bad_clip = range_ok([0.5, 0.5], [1.0, 1.0], [0.9, 1.6])
    assert not bad_clip["passed"] and bad_clip["by_p99"]
    assert not bad_clip["by_clip"]

    # --- ЧЕКПОЙНТ: ОТСУТСТВИЕ ПОЛЯ — ЭТО ОТКАЗ ---------------------------
    from types import SimpleNamespace
    ctx = SimpleNamespace(
        joint_sha="J", codec_fp={"codebooks_sha1": "c"},
        code_version={"bar.py": "b"}, decoder_context={"autocast": "off"},
        gate_info={"init_gate_sha1": "G"},
        q0_prov={"plan_sha1": "P", "q0_manifest_sha1": "M"})
    args = SimpleNamespace(variant="main", batch=8, dtype="float16",
                           q1_init="x")
    good = dict(joint_sha1="J", codec={"codebooks_sha1": "c"},
                code_version={"bar.py": "b"}, variant="main", batch=8,
                compute_dtype="float16", q1_init_sha1="Q",
                decoder_context={"autocast": "off"}, init_gate_sha1="G",
                q0_prov={"plan_sha1": "P", "q0_manifest_sha1": "M"},
                frozen_content_sha="F")
    assert set(good) == set(CHECKPOINT_FIELDS), sorted(
        set(good) ^ set(CHECKPOINT_FIELDS))
    assert check_checkpoint(good, ctx, args, lambda _p: "Q", "F") == []
    for key in sorted(good):
        broken = {k: v for k, v in good.items() if k != key}
        assert check_checkpoint(broken, ctx, args, lambda _p: "Q", "F"), (
            f"отсутствие {key} принято")
    assert any("frozen_content_sha" in p for p in check_checkpoint(
        dict(good, frozen_content_sha="ДРУГОЙ"), ctx, args,
        lambda _p: "Q", "F"))

    print("самопроверка k15b_probe_and_extract пройдена")


def main():
    ap = argparse.ArgumentParser(
        description="K-15b шаг 0: выбор книги C1 и цели читателя")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--checkpoint", default="data/k15/depth_rvq_s0.pt")
    ap.add_argument("--batches", type=int, default=0,
                    help="батчей val_sel; 0 — ВСЯ часть. Поднабор из 100 "
                         "батчей давал a0 на 8 % выше полной части, поэтому "
                         "для решения такого веса берём всё")
    ap.add_argument("--topk", type=int, default=10,
                    help="сколько РАЗЛИЧНЫХ ближайших кодов проверять")
    ap.add_argument("--tau-probe", type=float, default=1.0,
                    help="температура апостериора Q в диагностике; жёсткий "
                         "выбор от неё не зависит вовсе")
    ap.add_argument("--allow-smoke", action="store_true",
                    help="разрешить smoke-чекпойнт источником; книга при "
                         "этом не записывается ни при каком исходе")
    ap.add_argument("--out", default="reports/k15b/probe_and_extract.json")
    ap.add_argument("--out-c1", default="data/k15b/c1_selected.pt")
    ap.add_argument("--overwrite", action="store_true")
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import k15_context
    from k15_train_depth_rvq import H_EXEC
    k15_context.add_common_arguments(ap)
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if int(a.limit) != 0:
        raise SystemExit(
            f"--limit {a.limit}: probe решает про двухчасовой кэш и всё "
            f"последующее обучение, на урезанном плане это делать нельзя")
    for path in (a.out, a.out_c1):
        if os.path.exists(path) and not a.overwrite:
            raise SystemExit(f"{path} уже существует: без --overwrite не "
                             f"перезаписываю")

    ctx = k15_context.build(a)
    torch = ctx.torch
    dev, dt = ctx.dev, ctx.dt
    model, codec = ctx.model, ctx.codec
    names = list(ctx.info["names"])
    ac16 = torch.autocast(device_type=dev.type, dtype=dt)
    k15t = k15_context.k15t

    # ОТПЕЧАТОК ЗАМОРОЖЕННОГО СЧИТАЕТСЯ, А НЕ ОБЕЩАЕТСЯ В docstring.
    frozen_sha, n_frozen, n_elem = k15t.frozen_content_sha(
        model, torch, set(names))
    print(f"  замороженного: {n_frozen} тензоров, {n_elem} значений, "
          f"отпечаток {frozen_sha}")

    # --- ЧЕКПОЙНТ: FAIL-CLOSED -------------------------------------------
    if not os.path.exists(a.checkpoint):
        raise SystemExit(f"нет {a.checkpoint}")
    obj = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    kind = obj.get("kind")
    if kind not in ("k15_depth_rvq", "k15_smoke"):
        raise SystemExit(f"{a.checkpoint} описывает {kind!r}")
    if kind == "k15_smoke" and not a.allow_smoke:
        raise SystemExit(
            "это smoke-чекпойнт, его числа ничего не решают. Если нужен "
            "именно он — --allow-smoke; книга тогда не пишется")
    problems = check_checkpoint(obj, ctx, a, ctx.k11a.file_sha1, frozen_sha)
    if problems:
        raise SystemExit("чекпойнт K-15 собран в другой обстановке: "
                         + "; ".join(problems[:6]))
    print(f"  чекпойнт {a.checkpoint}: обстановка сверена по "
          f"{len(CHECKPOINT_FIELDS)} полям, включая побитовый отпечаток "
          f"замороженного")

    states = obj.get("states")
    if not isinstance(states, dict) or not states:
        raise SystemExit("в чекпойнте нет словаря states")
    if set(obj.get("trainable_names", [])) != set(names):
        raise SystemExit("белый список чекпойнта не совпал с текущим")
    own_shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    for tag, st in states.items():
        if set(st) != set(names):
            raise SystemExit(f"состояние {tag}: набор весов не тот")
        if obj.get(f"selected_state_sha1_{tag}") is None:
            raise SystemExit(
                f"состояние {tag}: нет selected_state_sha1_{tag}. "
                f"Отсутствие обязательного отпечатка — отказ, а не допуск")
        for k_, v_ in st.items():
            if tuple(v_.shape) != own_shapes[k_]:
                raise SystemExit(f"{tag}.{k_}: форма {tuple(v_.shape)} "
                                 f"против {own_shapes[k_]}")
            if not torch.isfinite(v_).all():
                raise SystemExit(f"{tag}.{k_}: нечисловые значения")

    # ДЕДУПЛИКАЦИЯ: q1 и q1_soft в прошлом прогоне оказались идентичны, и
    # треть работы ушла впустую.
    by_sha = {}
    for tag in sorted(states):
        by_sha.setdefault(obj[f"selected_state_sha1_{tag}"], []).append(tag)
    unique = {tags[0]: (sha, tags) for sha, tags in by_sha.items()}
    print(f"  состояний {sorted(states)}, уникальных по отпечатку "
          f"{len(unique)}: " + "; ".join(
              f"{t} = {'+'.join(tags)}" for t, (_s, tags) in unique.items()))

    batch_list = ctx.parts["val_sel"]
    take = strided(len(batch_list), a.batches)
    chosen_batches = [batch_list[i] for i in take]
    rows_all = np.unique(np.concatenate(
        [np.asarray(sel, np.int64) for _po, sel in chosen_batches]))
    rows_sha = hashlib.sha1(
        np.ascontiguousarray(rows_all).tobytes()).hexdigest()[:12]
    tasks = np.asarray(ctx.tsk)[rows_all]
    task_hist = {str(t): int(c) for t, c in
                 zip(*np.unique(tasks, return_counts=True))}
    print(f"  оценка на {len(chosen_batches)} батчах val_sel из "
          f"{len(batch_list)}, {rows_all.size} строк, отпечаток {rows_sha}")
    print(f"  задачи в поднаборе: {task_hist}")

    def restore(tag):
        with torch.no_grad():
            own = dict(model.state_dict())
            for k_, v_ in states[tag].items():
                own[k_].copy_(v_.to(own[k_].device, own[k_].dtype))
        got = ctx.k14c.state_sha(
            {k_: model.state_dict()[k_].detach().float().cpu().numpy()
             for k_ in names})
        want = obj[f"selected_state_sha1_{tag}"]
        if got != want:
            raise SystemExit(f"{tag}: после загрузки отпечаток {got}, в "
                             f"чекпойнте {want}")
        again, _n, _e = k15t.frozen_content_sha(model, torch, set(names))
        if again != frozen_sha:
            raise SystemExit(
                f"{tag}: отпечаток замороженного стал {again} против "
                f"{frozen_sha}: подстановка тронула не только белый список")
        return got

    PATHS = ("a0", "a1_tok", "a1_rank", "a1_pol", "a1_soft", "a2_tok",
             "a2_pol")
    TEACHER_PATHS = ("a1_tok", "a1_rank")

    def measure(tag):
        """Все числа одного состояния за один проход по выбранным батчам."""
        acc = {k: 0.0 for k in PATHS}
        n_rows = 0
        sup = {k: [] for k in TEACHER_PATHS + ("a1_pol", "a1_soft",
                                               "reference")}
        abs_vals = {k: [] for k in TEACHER_PATHS}
        agree = recall3 = recall10 = 0.0
        codes_q, codes_p, rank_choice = [], [], []
        rank_flips, first_batch = 0, [True]
        t0 = time.time()
        model.eval()
        with torch.no_grad():
            for po, sel in chosen_batches:
                b = ctx.build_batch(po, sel)
                am = b.get("attention_mask")
                with ac16:
                    v, p_ids = model.build_inputs(position_offset=po, **b)
                    out = model.forward_depth_aligned_rvq(
                        vlm_inputs_embeds=v, attention_mask=am,
                        position_ids=p_ids, mode="full", tau=1.0)
                q0_now = out["pred_codes"][0].detach().cpu().numpy()
                bad = int((q0_now != ctx.q0_can[sel]).sum())
                if bad:
                    raise SystemExit(
                        f"q0 разошёлся с каноническим в {bad} позициях: "
                        f"окружение собрано не так, как в тренере")
                z0 = out["policy_embeddings"][0]
                c1_pol = out["policy_embeddings"][1]
                c1 = model.depth_aligned_book(1)
                c2 = model.depth_aligned_book(2)
                action = torch.from_numpy(
                    np.asarray(ctx.ACT[sel], np.float32)).to(dev)[..., :7]
                z_e = codec._encode(action.float(), embodiment_ids=0).float()
                r1 = z_e - z0.detach()
                d1 = ctx.tok.mean_squared_distances(r1, c1)
                _lg, c1_tok, i1, _p1 = ctx.tok.quantize_residual(
                    r1, c1, temperature=float(a.tau_probe))
                hard_c1 = c1[out["pred_codes"][1]].detach()
                r2 = z_e - z0.detach() - hard_c1
                _lg2, c2_tok, _i2, _p2 = ctx.tok.quantize_residual(
                    r2, c2, temperature=float(a.tau_probe))
                soft1 = out["policy_probabilities"][1].float() @ c1.float()

                near = rank_candidates(d1, i1, int(a.topk), torch)
                if first_batch[0]:
                    srt = near.sort(dim=-1).values
                    if int((srt[..., 1:] == srt[..., :-1]).sum()):
                        raise SystemExit(
                            "кандидаты rank-path содержат дубли: проверялось "
                            "бы меньше кодов, чем заявлено")
                    print(f"      кандидатов на позицию {near.shape[-1]}, "
                          f"все различны")
                rank_flips += int(
                    ((-d1).topk(int(a.topk), dim=-1).indices[..., 0]
                     != i1).sum())
                errs = []
                for j in range(near.shape[-1]):
                    rows_j, _m = k15t.weighted_row_error(
                        ctx.decode_fp32(z0 + c1[near[..., j]]), action,
                        ctx.weights_gate, torch)
                    errs.append(rows_j)
                stack = torch.stack(errs, 0)
                best_j = stack.argmin(0)
                rank_choice.append(best_j.cpu().numpy())
                idx = best_j.view(-1, 1, 1).expand(-1, near.shape[1], 1)
                best_code = near.gather(-1, idx).squeeze(-1)

                lat = dict(
                    a0=z0, a1_tok=z0 + c1_tok, a1_rank=z0 + c1[best_code],
                    a1_pol=z0 + c1_pol, a1_soft=z0.float() + soft1,
                    a2_tok=z0 + hard_c1 + c2_tok,
                    a2_pol=out["cumulative_latents"][2])
                w = float(len(sel))
                n_rows += len(sel)
                acts, means = {}, {}
                for nm, z in lat.items():
                    acts[nm] = ctx.decode_fp32(z)
                    _rows, mean = k15t.weighted_row_error(
                        acts[nm], action, ctx.weights_gate, torch)
                    means[nm] = float(mean)
                    acc[nm] += means[nm] * w
                # ИНВАРИАНТ: ранг 1 — это и есть путь токенизатора, и он
                # проверяется на КАЖДОМ батче, а не на первом.
                r1_mean = float(stack[0].mean())
                if abs(r1_mean - means["a1_tok"]) / max(
                        means["a1_tok"], 1e-12) > 1e-9:
                    raise SystemExit(
                        f"ранг 1 дал {r1_mean!r}, путь токенизатора "
                        f"{means['a1_tok']!r}: кандидаты собраны не из того "
                        f"кода")
                first_batch[0] = False
                for nm in sup:
                    z = z_e if nm == "reference" else lat[nm]
                    sup[nm].append(k15t.decoder_support(
                        z.detach(), codec, torch, ctx.quantizers,
                        ctx.nearest_code, ctx.code_contribution)[1])
                for nm in TEACHER_PATHS:
                    abs_vals[nm].append(
                        acts[nm][:, :H_EXEC].abs().reshape(-1, 7)
                        .cpu().numpy())
                codes_q.append(i1[:, :H_EXEC].reshape(-1).cpu().numpy())
                codes_p.append(out["pred_codes"][1][:, :H_EXEC]
                               .reshape(-1).cpu().numpy())
                rd = k15t.reading_stats(
                    out["policy_probabilities"][1][:, :H_EXEC],
                    i1[:, :H_EXEC], torch,
                    policy_indices=out["pred_codes"][1][:, :H_EXEC])
                agree += rd["agree_top1"] * w
                recall3 += rd["recall_at3"] * w
                recall10 += rd["recall_at10"] * w
        elapsed = time.time() - t0
        n = max(n_rows, 1)
        res = {f"rms_{k}": float(np.sqrt(v / n)) for k, v in acc.items()}
        res.update(rows=n_rows, seconds=float(elapsed),
                   seconds_per_batch=float(
                       elapsed / max(len(chosen_batches), 1)),
                   rank1_flips_vs_topk=int(rank_flips))
        ref = torch.cat(sup["reference"])
        ref_p95 = float(torch.quantile(ref, 0.95))
        ref_p99 = float(torch.quantile(ref, 0.99))
        for nm in TEACHER_PATHS + ("a1_pol", "a1_soft"):
            v = torch.cat(sup[nm])
            res[f"support_{nm}"] = dict(
                mean=float(v.mean()), median=float(v.median()),
                p95=float(torch.quantile(v, 0.95)),
                p99=float(torch.quantile(v, 0.99)),
                reference_mean=float(ref.mean()), reference_p95=ref_p95,
                reference_p99=ref_p99,
                share_above_reference_p99=float((v > ref_p99).float().mean()),
                passed=bool(float(torch.quantile(v, 0.95)) <= ref_p95),
                rule="p95 остатка пути <= p95 остатка кодека на истинном "
                     "латенте")
        for nm in TEACHER_PATHS:
            flat = np.concatenate(abs_vals[nm], axis=0)
            p99 = [float(x) for x in np.percentile(flat, 99.0, axis=0)]
            amax = [float(x) for x in flat.max(axis=0)]
            res[f"range_{nm}"] = dict(
                p99_candidate=p99, absmax_candidate=amax,
                p99_dataset=[float(x) for x in ctx.act_p99_dataset],
                **range_ok(p99, ctx.act_p99_dataset, amax))
        book = model.depth_aligned_book(1).detach().float()
        norms = {int(i): float(x) for i, x in enumerate(
            book.norm(dim=-1).cpu().numpy())}
        set_q = set(np.concatenate(codes_q).tolist())
        set_p = set(np.concatenate(codes_p).tolist())

        def norm_stats(ids):
            if not ids:
                return dict(count=0, median=None, p95=None)
            v = np.asarray([norms[i] for i in ids], np.float64)
            return dict(count=len(ids), median=float(np.median(v)),
                        p95=float(np.percentile(v, 95)))

        res["row_sets"] = dict(
            used_by_q=len(set_q), used_by_p=len(set_p),
            jaccard=float(len(set_q & set_p) / max(len(set_q | set_p), 1)),
            both=norm_stats(sorted(set_q & set_p)),
            only_q=norm_stats(sorted(set_q - set_p)),
            only_p=norm_stats(sorted(set_p - set_q)),
            note=("нормы по РАЗЛИЧНЫМ строкам книги, не взвешенные частотой "
                  "использования. Это НЕ та же статистика, что медиана по "
                  "позициям данных, и одна другую не предсказывает"))
        res["agree_top1"] = agree / n
        res["recall_at3"] = recall3 / n
        res["recall_at10"] = recall10 / n
        rc = np.concatenate(rank_choice)
        res["rank_choice"] = dict(
            share_rank1=float((rc == 0).mean()),
            share_changed=float((rc != 0).mean()),
            mean_rank=float(rc.mean()),
            histogram={int(k): int(v) for k, v in
                       zip(*np.unique(rc, return_counts=True))})
        res["relative_gain_rank_over_latent"] = float(
            (res["rms_a1_tok"] - res["rms_a1_rank"])
            / max(res["rms_a1_tok"], 1e-12))
        res["fraction_of_gap_rank"] = fraction_of_gap(
            res["rms_a0"], res["rms_a1_tok"], res["rms_a1_rank"])
        res["fraction_of_gap_pol"] = fraction_of_gap(
            res["rms_a0"], res["rms_a1_tok"], res["rms_a1_pol"])
        return res

    # --- ИЗМЕРЕНИЕ ВСЕХ УНИКАЛЬНЫХ СОСТОЯНИЙ -----------------------------
    per_state = {}
    for tag, (sha, aliases) in unique.items():
        print(f"\n  состояние {tag} (эпоха "
              f"{obj.get(f'selected_epoch_{tag}')}, {sha}, совпадает с "
              f"{aliases})")
        restore(tag)
        r = measure(tag)
        r.update(state_sha1=sha, aliases=aliases,
                 epoch=obj.get(f"selected_epoch_{tag}"))
        per_state[tag] = r
        print(f"    a0 {r['rms_a0']:.6f}; латентная цель a1_tok "
              f"{r['rms_a1_tok']:.6f}; rank-path a1_rank "
              f"{r['rms_a1_rank']:.6f} (выигрыш "
              f"{100 * r['relative_gain_rank_over_latent']:.1f} %); жёсткий "
              f"a1_pol {r['rms_a1_pol']:.6f}; мягкий {r['rms_a1_soft']:.6f}")
        for nm in TEACHER_PATHS:
            s, g = r[f"support_{nm}"], r[f"range_{nm}"]
            print(f"    {nm}: опора p95 {s['p95']:.4f} при эталонном "
                  f"{s['reference_p95']:.4f} — "
                  f"{'пройдено' if s['passed'] else 'ОТКАЗ'}; диапазон "
                  f"{'пройден' if g['passed'] else 'ОТКАЗ'} (p99 "
                  f"{'ok' if g['by_p99'] else 'нет'}, предел шкалы "
                  f"{'ok' if g['by_clip'] else 'нет'})")
        rs = r["row_sets"]
        print(f"    строки книги: Q {rs['used_by_q']}, P {rs['used_by_p']}, "
              f"Jaccard {rs['jaccard']:.3f}; медианы норм — оба "
              f"{rs['both']['median']}, только Q {rs['only_q']['median']}, "
              f"только P {rs['only_p']['median']}")
        print(f"    чтение: согласие {100 * r['agree_top1']:.2f} %, recall@3 "
              f"{100 * r['recall_at3']:.1f} %, recall@10 "
              f"{100 * r['recall_at10']:.1f} %; выбран не ранг 1 на "
              f"{100 * r['rank_choice']['share_changed']:.1f} % строк")
        print(f"    {r['seconds_per_batch']:.2f} с/батч")

    # --- ВЫБОР: СНАЧАЛА ЦЕЛЬ КНИГИ, ПОТОМ ГЕЙТЫ ЕЁ ПУТИ -----------------
    per_batch = max(r["seconds_per_batch"] for r in per_state.values())
    cache_hours = per_batch * len(ctx.parts_full["train"]) / 3600.0
    table = {}
    for tag, r in per_state.items():
        dec = target_decision(r["relative_gain_rank_over_latent"],
                              cache_hours)
        path = target_path_name(dec["target"])
        sup_ok = r[f"support_{path}"]["passed"]
        rng_ok = r[f"range_{path}"]["passed"]
        table[tag] = dict(
            epoch=r["epoch"], state_sha1=r["state_sha1"],
            target=dec["target"], teacher_path=path,
            teacher_rms=float(r[f"rms_{path}"]),
            support_passed=bool(sup_ok), range_passed=bool(rng_ok),
            admissible=bool(sup_ok and rng_ok), decision=dec,
            rms_latent=float(r["rms_a1_tok"]),
            rms_rankpath=float(r["rms_a1_rank"]),
            support_latent_p95=r["support_a1_tok"]["p95"],
            support_rankpath_p95=r["support_a1_rank"]["p95"],
            support_reference_p95=r["support_a1_tok"]["reference_p95"])
    print("\n  ТАБЛИЦА РЕШЕНИЯ (цель определяется для каждой книги "
          "отдельно, гейты — для пути её цели):")
    print(f"    {'книга':>10} {'эп':>3} {'цель':>22} {'RMS цели':>9} "
          f"{'опора':>7} {'диапазон':>9} {'допустима':>10}")
    for tag, t in table.items():
        print(f"    {tag:>10} {str(t['epoch']):>3} {t['target']:>22} "
              f"{t['teacher_rms']:9.6f} "
              f"{('ok' if t['support_passed'] else 'ОТКАЗ'):>7} "
              f"{('ok' if t['range_passed'] else 'ОТКАЗ'):>9} "
              f"{('да' if t['admissible'] else 'НЕТ'):>10}")
    print(f"    полный кэш train оценён в {cache_hours:.1f} ч при бюджете "
          f"{CACHE_BUDGET_HOURS:.1f} ч")

    admissible = {t: v for t, v in table.items() if v["admissible"]}
    unconstrained = min(table, key=lambda t: (
        round(table[t]["teacher_rms"], 12), t))
    best = (min(admissible, key=lambda t: (
        round(admissible[t]["teacher_rms"], 12), t)) if admissible else None)
    selection = dict(
        rule=("минимум RMS УЧИТЕЛЬСКОГО пути среди книг, прошедших опору и "
              "диапазон этого же пути"),
        rule_changed_after_data=(
            "ДА: исходное правило было «минимум a1_tok». Ограничение по "
            "опоре и привязка гейтов к пути цели добавлены после того, как "
            "книга эпохи 2 опору не прошла"),
        admissible=sorted(admissible), selected=best,
        unconstrained_best=unconstrained,
        differs_from_unconstrained=bool(best != unconstrained))
    if best is None:
        print("\n  ДОПУСТИМОЙ КНИГИ НЕТ: ни один учительский путь не "
              "проходит опору и диапазон. Книга не записывается.")
    else:
        print(f"\n  ВЫБРАНА КНИГА {best} (эпоха {table[best]['epoch']}), "
              f"цель {table[best]['target']}, RMS учительского пути "
              f"{table[best]['teacher_rms']:.6f}")
        if best != unconstrained:
            print(f"    безусловный минимум был бы {unconstrained} "
                  f"({table[unconstrained]['teacher_rms']:.6f}), но его путь "
                  f"не проходит гейты")

    # --- ЗАПИСЬ КНИГИ: ТОЛЬКО ПОСЛЕ ВСЕХ ГЕЙТОВ --------------------------
    c1_written, c1_sha = None, None
    if best is not None and kind != "k15_smoke":
        restore(best)
        c1_sel = model.depth_aligned_book(1).detach().cpu().clone()
        c1_sha = hashlib.sha1(np.ascontiguousarray(
            c1_sel.float().numpy()).tobytes()).hexdigest()[:12]
        tp = table[best]["teacher_path"]
        payload = dict(
            kind="k15b_c1_selected", accepted=True,
            c1=c1_sel, c1_sha1=c1_sha, c1_shape=list(c1_sel.shape),
            source_checkpoint=os.path.abspath(a.checkpoint),
            source_checkpoint_sha1=k15_context.sha12(a.checkpoint),
            source_state=best, source_aliases=per_state[best]["aliases"],
            source_epoch=table[best]["epoch"],
            source_state_sha1=table[best]["state_sha1"],
            teacher_target=table[best]["target"], teacher_path=tp,
            teacher_rms=table[best]["teacher_rms"],
            support=per_state[best][f"support_{tp}"],
            action_range=per_state[best][f"range_{tp}"],
            topk=int(a.topk), selection=selection,
            rows=int(rows_all.size), rows_sha1=rows_sha,
            task_histogram=task_hist, frozen_content_sha=frozen_sha,
            codec=ctx.codec_fp, code_version=ctx.code_version,
            q0_prov=ctx.q0_prov, joint_sha1=ctx.joint_sha,
            git_head=ctx.git_head, git_dirty=bool(ctx.dirty),
            **ctx.gate_info)
        os.makedirs(os.path.dirname(os.path.abspath(a.out_c1)) or ".",
                    exist_ok=True)
        tmp = a.out_c1 + f".tmp.{os.getpid()}"
        torch.save(payload, tmp)
        os.replace(tmp, a.out_c1)
        back = torch.load(a.out_c1, map_location="cpu", weights_only=False)
        if not torch.equal(back["c1"], c1_sel) or back["c1_sha1"] != c1_sha:
            raise SystemExit("книга после обратного чтения отличается")
        if back.get("accepted") is not True:
            raise SystemExit("в файле книги нет accepted=True")
        c1_written = os.path.abspath(a.out_c1)
        print(f"  книга сохранена: {a.out_c1} ({c1_sha}), accepted=True")
    elif kind == "k15_smoke":
        print("  smoke-источник: книга не записывается ни при каком исходе")

    own_keys = {"kind", "checkpoint", "checkpoint_kind", "batches",
                "batches_in_part", "rows", "rows_sha1", "task_histogram",
                "tau_probe", "topk", "per_state", "table", "selection",
                "cache_hours", "c1_file", "c1_sha1", "codec", "code_version",
                "q0_prov", "joint_sha1", "git_head", "git_dirty",
                "frozen_content_sha", "target_rule", "cache_budget_hours",
                "not_a_full_oracle", "rankpath_definition"}
    clash = sorted(set(ctx.gate_info) & own_keys)
    if clash:
        raise SystemExit(f"ключи {clash} из gate_info сталкиваются с полями "
                         f"отчёта")
    report = dict(
        kind="k15b_probe_and_extract",
        checkpoint=os.path.abspath(a.checkpoint), checkpoint_kind=kind,
        batches=len(chosen_batches), batches_in_part=len(batch_list),
        rows=int(rows_all.size), rows_sha1=rows_sha,
        task_histogram=task_hist, tau_probe=float(a.tau_probe),
        topk=int(a.topk),
        rankpath_definition=(
            "траектория j берёт j-й по близости код НА КАЖДОЙ позиции чанка; "
            "по строке выбирается лучший НОМЕР РАНГА. Кандидаты на позицию "
            "различны, первый — исполняемый код модели"),
        not_a_full_oracle=(
            "§50.2 этим НЕ закрыт по двум причинам: лучший по действию код "
            "может лежать вне топ-k ближайших латентных строк, и смеси "
            "рангов по позициям (k^T вариантов) не перебираются вовсе"),
        per_state=per_state, table=table, selection=selection,
        cache_hours=float(cache_hours),
        cache_budget_hours=CACHE_BUDGET_HOURS,
        target_rule=[dict(below=t if np.isfinite(t) else None, choice=c,
                          why=w) for t, c, w in TARGET_RULE],
        c1_file=c1_written, c1_sha1=c1_sha,
        frozen_content_sha=frozen_sha,
        codec=ctx.codec_fp, code_version=ctx.code_version,
        q0_prov=ctx.q0_prov, joint_sha1=ctx.joint_sha,
        git_head=ctx.git_head, git_dirty=bool(ctx.dirty), **ctx.gate_info)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    tmp = a.out + f".tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1,
                  default=k15t.json_scalar)
    os.replace(tmp, a.out)
    print(f"  отчёт: {a.out}")

    if best is None:
        print("\n  РЕШЕНИЕ: допустимой книги нет. По дереву остаётся "
              "восстановить эпоху 1 (2.2 ч обучения) и проверить, есть ли "
              "точка между качеством и опорой.")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
