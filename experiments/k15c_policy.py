#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15c: рука-селектор для поведенческого харнесса k9h_multiarm_gate.

ЧТО ИСПОЛНЯЕТСЯ НА КАЖДОМ ВЫЗОВЕ ПОЛИТИКИ — ровно путь развёртывания:

    один проход VLA `mode="full"` (24 слоя), pre-hook снимает h24,
    устойчивый порядок кодов q1, восемь согласованных путей,
    голова выбирает один, z = z0 + C1[выбранные коды],
    ОДИН декод этого z (декодирует харнесс, получая `Latent`).

Выбор делает `k15c_rank_selector.select_from_outputs` — та же функция, что в
проверке вывода. Поэтому рука принимается только вместе с отчётом проверки
вывода, который (а) прошёл, (б) снят для ТОГО ЖЕ файла головы и (в) ТЕМ ЖЕ
модулем голов. Отчёт старой версии, без отпечатка модуля, не принимается:
тогда совпадение выбора в роллауте и в проверке ничем не доказано.

ОБСТАНОВКА — каноническая K-15b: `k15c_build_rank_cache.load_stack`, то
есть те же модель, книга, читатель e0s0, кодек и гейт K-15a, что у кэша и у
обучения головы. Гейт пинит карту, поэтому рука работает только на той же
карте, что и гейт.

МОДЕЛЬ СОЗДАЁТСЯ ПОСЛЕ СРЕД. Харнесс поднимает среды до инициализации CUDA
(fork после неё вешает процесс), и рука собирается уже после `get_envs`.
Сиды не страдают: раскатка пересевается перед каждым раундом.

СТАТИСТИКА ВЫБОРА ведётся по блоку и уходит в его артефакт: какие ранги
выбирались и сколько было вызовов. Это и есть «распределение выбранных
рангов» поведенческого отчёта.
"""
import argparse
import hashlib
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

ARM_KIND = "k15c_rank_selector"
REPORT_KIND = "k15c_inference_check"


def check_binding(sel_obj, sel_path, sel_sha, man, man_sha, report,
                  module_sha, stack, preflight=False):
    """Голова, кэш, отчёт проверки вывода и обстановка — одно целое.

    Чистая функция; `stack` — словарь c1_sha1, reader_state_sha1,
    frozen_content_sha, codec, plan_sha1, decoder_context, code_version,
    joint_sha1. Возвращает список проблем.
    """
    p = []

    def need(cond, msg):
        if not cond:
            p.append(msg)

    need(sel_obj.get("kind") == ARM_KIND, f"голова: kind "
         f"{sel_obj.get('kind')!r}")
    if preflight:
        # ПРЕДПОЛЁТНАЯ ПРОВЕРКА: допускается ТОЛЬКО smoke-голова на
        # smoke-кэше со smoke-отчётом проверки вывода, чтобы настоящий
        # результат нельзя было получить этим путём ни по ошибке, ни
        # намеренно. Связь «отчёт -> рука» проверяется ТЕМИ ЖЕ условиями,
        # что ночью: иначе предполётная проверка не покрывала бы контракт,
        # на котором держится настоящий запуск.
        need(sel_obj.get("smoke") is True,
             "предполётная проверка — только для smoke-головы")
        need(man.get("canonical") is False,
             "предполётная проверка — только на smoke-кэше")
    else:
        need(not sel_obj.get("smoke"), "голова обучена на smoke-кэше")
    need(str(sel_obj.get("head", "")).startswith("h24_"),
         f"голова {sel_obj.get('head')!r}: развёртывается только h24")
    need(not str(sel_path).endswith(".technical_fail.pt"),
         "чекпойнт головы с техническим отказом")
    need(sel_obj.get("cache_manifest_sha1") == man_sha,
         "голова обучалась на другом кэше")
    if not preflight:
        need(man.get("canonical") is True, "кэш не канонический")
    for key in ("c1_sha1", "reader_state_sha1"):
        need(sel_obj.get(key) == man.get(key) == stack.get(key),
             f"{key}: голова {sel_obj.get(key)!r}, кэш {man.get(key)!r}, "
             f"обстановка {stack.get(key)!r}")
    need(man.get("frozen_content_sha") == stack.get("frozen_content_sha"),
         "замороженное не то, что при построении кэша")
    need(man.get("codec") == stack.get("codec"), "кодек не тот")
    for key in ("plan_sha1", "decoder_context", "code_version",
                "joint_sha1"):
        need(man.get(key) == stack.get(key),
             f"{key}: в кэше {man.get(key)!r}, в обстановке "
             f"{stack.get(key)!r}")
    if report is None:
        p.append("нет отчёта проверки вывода")
        return p
    if preflight:
        need(report.get("smoke") is True,
             "предполётная проверка — только со smoke-отчётом")
    else:
        need(report.get("smoke") is not True,
             "отчёт проверки вывода снят на smoke-кэше")
    need(report.get("kind") == REPORT_KIND,
         f"отчёт: kind {report.get('kind')!r}")
    need((report.get("verdict") or {}).get("passed") is True,
         "проверка вывода не пройдена")
    need(report.get("selector_file_sha1") == sel_sha,
         "отчёт снят для другого файла головы")
    need(report.get("selector_module_sha1") is not None,
         "отчёт снят старой версией, без отпечатка модуля голов: совпадение "
         "выбора в роллауте и в проверке ничем не доказано — переснимите "
         "проверку вывода")
    need(report.get("selector_module_sha1") in (None, module_sha),
         "отчёт снят другой версией модуля голов")
    need(report.get("cache_manifest_sha1") == man_sha,
         "отчёт снят на другом кэше")
    # СТАТУС ГОЛОВЫ ПРИВЯЗАН К ОТЧЁТУ: роллаут положен только голове,
    # взявшей основной или разведочный порог. Режим роллаута (dev только
    # для основного) проверяет раннер по этому же полю.
    allowed = ("primary", "pilot") + (("smoke",) if preflight else ())
    need(report.get("selector_status") in allowed,
         f"статус головы {report.get('selector_status')!r}: роллаут положен "
         f"только при основном или разведочном пороге")
    return p


class PickStats:
    """Счётчик выбранных рангов по блоку. take() отдаёт и обнуляет."""

    def __init__(self, k=8):
        self.k = int(k)
        self.reset()

    def reset(self):
        self.hist = np.zeros(self.k, np.int64)
        self.calls = 0

    def add(self, picks):
        a = np.asarray(picks, np.int64).ravel()
        if a.size and (a.min() < 0 or a.max() >= self.k):
            raise ValueError(f"ранг вне [0, {self.k})")
        self.hist += np.bincount(a, minlength=self.k)
        self.calls += 1

    def take(self):
        n = int(self.hist.sum())
        out = dict(calls=int(self.calls), choices=n,
                   histogram={int(i): int(c) for i, c in
                              enumerate(self.hist)},
                   share_rank0=(float(self.hist[0] / n) if n else None))
        self.reset()
        return out


def stack_namespace(device, root="third_party/actioncodec"):
    """Аргументы `load_stack` — канонические умолчания K-15b и K-15c."""
    import k15_context
    ns = dict(k15_context.DEFAULTS)
    ns.update(device=str(device), root=root,
              c1="data/k15b/c1_selected.pt",
              k15b_target="data/k15b/rankpath_target_train.npz",
              checkpoint="data/k15b/q1_reader_s0.pt")
    return argparse.Namespace(**ns)


def build_arm(device, selector, rank_cache, inference_report, torch,
              preflight=False):
    """Рука целиком: модель, голова, хуки, привязка. Отказ — с текстом.

    preflight=True — проверка МЕХАНИКИ роллаута на smoke-голове и
    smoke-кэше: отчёт проверки вывода не нужен, но настоящая голова и
    канонический кэш здесь отвергаются, а артефакты помечаются.
    """
    import k15c_build_rank_cache as cb
    import k15c_rank_selector as rs

    for f in (selector, os.path.join(rank_cache, "manifest.json"),
              inference_report):
        if not os.path.exists(f):
            raise SystemExit(f"нет {f}")
    sel_obj = torch.load(selector, map_location="cpu", weights_only=False)
    # Файлы кэша в роллауте не читаются, нужен только манифест: отпечатки
    # массивов здесь не пересчитываются, формы и COMPLETE — проверяются.
    man, problems = cb.validate_cache(rank_cache, verify_sha=False,
                                      allow_smoke=bool(preflight))
    if problems:
        raise SystemExit("кэш не принят: " + "; ".join(problems[:6]))
    man_sha = cb.sha_file(os.path.join(rank_cache, "manifest.json"))
    with open(inference_report, encoding="utf-8") as fh:
        report = json.load(fh)

    S = cb.load_stack(stack_namespace(device))
    # ЧЕРЕЗ JSON: в манифесте поля прошли сериализацию, и сравнивать надо
    # то же представление, иначе кортеж против списка дал бы ложный отказ.
    stack = json.loads(json.dumps(dict(
        c1_sha1=S.book["c1_sha1"], reader_state_sha1=S.point_sha,
        frozen_content_sha=S.frozen_sha, codec=S.ctx.codec_fp,
        plan_sha1=S.ctx.q0_prov.get("plan_sha1"),
        decoder_context=S.ctx.decoder_context,
        code_version=S.ctx.code_version, joint_sha1=S.ctx.joint_sha),
        default=str))
    module_sha = cb.sha_file(rs.__file__)
    sel_sha = cb.sha_file(selector)
    problems = check_binding(sel_obj, selector, sel_sha, man, man_sha,
                             report, module_sha, stack, preflight=preflight)
    if problems:
        raise SystemExit("рука K-15c не собрана: " + "; ".join(problems))

    model = S.model
    c1 = model.depth_aligned_book(1)
    head = rs.build_head(sel_obj["head"], int(sel_obj["d_model"]),
                         int(sel_obj["e_dim"]), torch,
                         proj=int(sel_obj["proj"]),
                         book=c1.detach().float()).to(S.ctx.dev)
    head.load_state_dict(sel_obj["state"])
    head.eval()
    hooks = cb.StateHooks(model)
    stats = PickStats()
    meta = dict(
        arm="k15c", selector=os.path.abspath(selector),
        selector_status=("preflight" if preflight
                         else report.get("selector_status")),
        inference_report_smoke=bool(report.get("smoke")),
        preflight=bool(preflight),
        selector_file_sha1=sel_sha, head=sel_obj["head"],
        selected_epoch=sel_obj.get("selected_epoch"),
        selector_module_sha1=module_sha,
        inference_report=os.path.abspath(inference_report),
        inference_report_sha1=cb.sha_file(inference_report),
        rank_cache_manifest_sha1=man_sha, c1_sha1=stack["c1_sha1"],
        reader_state_sha1=stack["reader_state_sha1"],
        frozen_content_sha=stack["frozen_content_sha"],
        codec=stack["codec"], gate=S.ctx.gate_info,
        code_version=S.ctx.code_version, joint_sha1=S.ctx.joint_sha,
        h24_feedback_note=man.get("h24_feedback_note"),
        decodes_per_call=1, layers_per_call=24)
    # МЕТАДАННЫЕ ПРОВЕРЯЮТСЯ НА СЕРИАЛИЗУЕМОСТЬ СРАЗУ. Харнесс пишет их в
    # артефакт блока через json.dump без обработчика типов, и неописанный
    # тип уронил бы блок ПОСЛЕ раскатки — её результат пропал бы.
    import k15_train_depth_rvq as k15t
    try:
        meta = json.loads(json.dumps(meta, default=k15t.json_scalar))
    except TypeError as e:
        raise SystemExit(f"метаданные руки не сериализуются: {e}")
    meta["model_fingerprint"] = hashlib.sha1("|".join(
        str(meta[k]) for k in ("selector_file_sha1", "selector_module_sha1",
                               "c1_sha1", "reader_state_sha1",
                               "frozen_content_sha", "joint_sha1")
    ).encode()).hexdigest()[:12]

    def act(batch, pos_off, autocast, first):
        """(z выбранного пути, коды q0) за один проход и один декод."""
        with torch.no_grad(), autocast:
            v, p_ = model.build_inputs(position_offset=pos_off, **batch)
            out = model.forward_depth_aligned_rvq(
                vlm_inputs_embeds=v,
                attention_mask=batch.get("attention_mask"),
                position_ids=p_, mode="full", tau=1.0)
        _h18, h24 = hooks.take()
        with torch.no_grad():
            try:
                sel = rs.select_from_outputs(out["logits"][1],
                                             out["policy_embeddings"][0],
                                             h24, head, c1, torch)
            except ValueError as e:
                raise SystemExit(f"рука k15c: {e}")
        z = sel["z"]
        if not bool(torch.isfinite(z).all()):
            raise SystemExit("латент выбранного пути не конечен")
        if first:
            if int(out["layers_run"]) != 24:
                raise SystemExit(f"проход {out['layers_run']} слоёв вместо "
                                 f"24: это не один полный проход")
            print(f"    проверка k15c: слоёв {out['layers_run']}, голова "
                  f"{sel_obj['head']}, ранги первого вызова "
                  f"{sel['pick'].tolist()}", flush=True)
        stats.add(sel["pick"].cpu().numpy())
        return z, out["pred_codes"][0].cpu().numpy()

    return SimpleNamespace(model=model, proc=S.ctx.proc, codec=S.ctx.codec,
                           act=act, stats=stats, meta=meta, stack=S)


def selftest():
    # --- ПРИВЯЗКА ГОЛОВЫ, КЭША, ОТЧЁТА И ОБСТАНОВКИ ---------------------
    env = dict(plan_sha1="P", decoder_context={"autocast": "disabled"},
               code_version={"bar.py": "b"}, joint_sha1="J")
    stack = dict(c1_sha1="C", reader_state_sha1="R", frozen_content_sha="F",
                 codec={"c": 1}, **env)
    man = dict(canonical=True, c1_sha1="C", reader_state_sha1="R",
               frozen_content_sha="F", codec={"c": 1}, **env)
    sel = dict(kind=ARM_KIND, smoke=False, head="h24_candidate",
               cache_manifest_sha1="M", c1_sha1="C", reader_state_sha1="R")
    rep = dict(kind=REPORT_KIND, verdict=dict(passed=True),
               selector_file_sha1="S", selector_module_sha1="MOD",
               cache_manifest_sha1="M", selector_status="primary")
    ok = check_binding(sel, "x/h24_candidate_s0.pt", "S", man, "M", rep,
                       "MOD", stack)
    assert ok == [], ok
    muts = [
        ("голова smoke", dict(sel=dict(sel, smoke=True))),
        ("голова h18", dict(sel=dict(sel, head="h18_linear"))),
        ("другой кэш у головы", dict(sel=dict(sel,
                                              cache_manifest_sha1="X"))),
        ("кэш smoke", dict(man=dict(man, canonical=False))),
        ("другая книга", dict(stack=dict(stack, c1_sha1="X"))),
        ("другой читатель", dict(man=dict(man, reader_state_sha1="X"))),
        ("другое замороженное", dict(stack=dict(stack,
                                                frozen_content_sha="X"))),
        ("другой кодек", dict(stack=dict(stack, codec={"c": 2}))),
        ("отчёт не пройден", dict(rep=dict(rep, verdict=dict(
            passed=False)))),
        ("отчёт другой головы", dict(rep=dict(rep,
                                              selector_file_sha1="X"))),
        ("отчёт старой версии", dict(rep={k: v for k, v in rep.items()
                                          if k != "selector_module_sha1"})),
        ("отчёт другим модулем", dict(module="ИНОЙ")),
        ("отчёт на другом кэше", dict(rep=dict(rep,
                                               cache_manifest_sha1="X"))),
        ("нет отчёта", dict(rep=None)),
        ("technical_fail", dict(path="x/h.technical_fail.pt")),
        ("другой план", dict(stack=dict(stack, plan_sha1="X"))),
        ("другой контекст декодера", dict(man=dict(
            man, decoder_context={"autocast": "enabled"}))),
        ("другая версия архитектуры", dict(stack=dict(
            stack, code_version={"bar.py": "x"}))),
        ("другой joint", dict(man=dict(man, joint_sha1="X"))),
        ("голова без порога", dict(rep=dict(rep, selector_status=None))),
        ("статус выдуман", dict(rep=dict(rep, selector_status="none"))),
    ]
    # ОТЧЁТ, СНЯТЫЙ НА SMOKE-КЭШЕ, НАСТОЯЩЕЙ РУКОЙ НЕ ПРИНИМАЕТСЯ
    assert check_binding(sel, "x/h.pt", "S", man, "M", dict(rep, smoke=True),
                         "MOD", stack)
    # ПРЕДПОЛЁТНЫЙ РЕЖИМ: smoke-голова, smoke-кэш и smoke-ОТЧЁТ, связанные
    # теми же условиями, что настоящий запуск
    sm_sel, sm_man = dict(sel, smoke=True), dict(man, canonical=False)
    sm_rep = dict(rep, smoke=True, selector_status="smoke")
    assert check_binding(sm_sel, "x/h.pt", "S", sm_man, "M", sm_rep, "MOD",
                         stack, preflight=True) == []
    for why, kw in (
            ("настоящая голова", dict(sel_=sel)),
            ("канонический кэш", dict(man_=man)),
            ("нет отчёта", dict(rep_=None)),
            ("настоящий отчёт", dict(rep_=dict(rep))),
            ("отчёт другой головы", dict(rep_=dict(sm_rep,
                                                   selector_file_sha1="X"))),
            ("отчёт другим модулем", dict(module_="ИНОЙ")),
            ("отчёт на другом кэше", dict(rep_=dict(sm_rep,
                                                    cache_manifest_sha1="X"))),
            ("проверка не пройдена", dict(rep_=dict(sm_rep, verdict=dict(
                passed=False)))),
            ("другая обстановка", dict(stack_=dict(stack, plan_sha1="X")))):
        got = check_binding(kw.get("sel_", sm_sel), "x/h.pt", "S",
                            kw.get("man_", sm_man), "M",
                            kw.get("rep_", sm_rep) if "rep_" in kw
                            else sm_rep, kw.get("module_", "MOD"),
                            kw.get("stack_", stack), preflight=True)
        assert got, f"предполётная мутация «{why}» не поймана"
    # статус smoke вне предполётного режима не принимается
    assert check_binding(sel, "x/h.pt", "S", man, "M",
                         dict(rep, selector_status="smoke"), "MOD", stack)
    # РАЗВЕДОЧНЫЙ СТАТУС ДЛЯ РУКИ ДОПУСТИМ — ограничение режима у раннера
    assert check_binding(sel, "x/h.pt", "S", man, "M",
                         dict(rep, selector_status="pilot"), "MOD",
                         stack) == []
    for why, m in muts:
        got = check_binding(m.get("sel", sel), m.get("path", "x/h.pt"), "S",
                            m.get("man", man), "M", m.get("rep", rep),
                            m.get("module", "MOD"), m.get("stack", stack))
        assert got, f"мутация «{why}» не поймана"

    # --- СТАТИСТИКА ВЫБОРА ПО БЛОКУ -------------------------------------
    st = PickStats()
    st.add([0, 3, 3])
    st.add([7, 0])
    t = st.take()
    assert t["calls"] == 2 and t["choices"] == 5, t
    assert t["histogram"][3] == 2 and t["histogram"][0] == 2, t
    assert abs(t["share_rank0"] - 0.4) < 1e-12
    assert st.take()["choices"] == 0 and st.take()["share_rank0"] is None
    try:
        st.add([8])
    except ValueError as e:
        assert "вне" in str(e), e
    else:
        raise AssertionError("принят ранг 8")
    print(f"самопроверка k15c_policy пройдена: привязка с {len(muts)} "
          f"мутациями, статистика выбора по блоку")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="K-15c: рука-селектор")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        selftest()
        sys.exit(0)
    raise SystemExit("это модуль руки для k9h_multiarm_gate --policy k15c")
