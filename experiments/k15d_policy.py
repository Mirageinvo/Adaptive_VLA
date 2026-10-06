#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15d: рука уточнения D0 или H1 для поведенческого харнесса k9h.

ЧТО ИСПОЛНЯЕТСЯ НА КАЖДОМ ВЫЗОВЕ ПОЛИТИКИ — ровно путь обучения и оценки:

    один проход `DepthRefiner.run` (D0: 24 слоя; H1: 24 слоя, a1 на 18),
    a0 = декод канонического q0, итоговое действие — последний уровень
    (D0: a_d, H1: a2). Харнесс получает ГОТОВЫЙ нормированный чанк
    (`ActionChunk`) и декодер кодека к нему не применяет.

ПРИВЯЗКА. Рука собирается только если:
  * чекпойнт — завершённый (`status == "complete"`, `final`,
    `technical_ok`), фазы d0 или h1p2, режима full, допущен фильтром
    (`admission.admissible`); в предполётном режиме — наоборот, только
    smoke-чекпойнт;
  * гейт K-15d ЭТОЙ карты пройден с тем же кодом уточнения, статистикой,
    замороженным и гиперпараметрами, что у чекпойнта;
  * отчёт проверки вывода снят для ЭТОГО файла чекпойнта, ЭТИМ модулем
    уточнения и ЭТИМ модулем руки, и пройден.

ПЕРВЫЙ ВЫЗОВ проверяет, что итоговый уровень действительно отошёл от a0:
нулевая поправка значила бы, что веса уточнения не встали, и под меткой d0
или h1 исполнялся бы q0.

ЗАПИСЬ УРОВНЕЙ. На каждом вызове сохраняются исполняемые шаги a0 и всех
уровней; харнесс кладёт их в npz действий блока (план §13: «запись
a0, a1, a2»). Сводка по блоку — средний |поправки| по каналам и максимум |a|.
"""
import argparse
import hashlib
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARM_PHASES = {"d0": "d", "h1p2": "2"}
REPORT_KIND = "k15d_inference_check"
ACTION_CLIP_BOUND = 1.5


def sha_file(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()[:12]


def gate_paths(device):
    """Гейты K-15a и K-15d для карты: cuda:1 — канонические, иначе свои."""
    if str(device) == "cuda:1":
        return ("reports/k15a/init_identity.json",
                "reports/k15d/init_identity.json")
    tag = str(device).replace(":", "")
    return (f"reports/k15a/init_identity_{tag}.json",
            f"reports/k15d/init_identity_{tag}.json")


def check_checkpoint(ck, *, preflight=False):
    """Чекпойнт годен для руки. Чистая; список проблем."""
    p = []

    def need(cond, msg):
        if not cond:
            p.append(msg)

    need(ck.get("kind") == "k15d_phase", f"kind {ck.get('kind')!r}")
    need(ck.get("phase") in ARM_PHASES,
         f"фаза {ck.get('phase')!r}: рука — только d0 или h1p2")
    need(ck.get("status") == "complete" and ck.get("final") is True,
         f"чекпойнт не завершён: status {ck.get('status')!r}")
    need(ck.get("technical_ok") is True, "технически не исправен")
    need(bool(ck.get("selected_tag")), "нет выбранной точки")
    if preflight:
        need(ck.get("mode") == "smoke",
             "предполётная проверка — только smoke-чекпойнт")
    else:
        need(ck.get("mode") == "full", f"режим {ck.get('mode')!r}, нужен full")
        need((ck.get("admission") or {}).get("admissible") is True,
             "фильтр допуска к роллауту не пройден")
    return p


def check_gate_vs_ckpt(gate, ck, *, device):
    """Гейт карты и чекпойнт описывают одну и ту же обстановку."""
    p = []
    if gate.get("device") != str(device):
        p.append(f"гейт снят на {gate.get('device')}, рука на {device}")
    for key in ("stats_sha1", "frozen_sha1"):
        if gate.get(key) != ck.get(key):
            p.append(f"{key}: гейт {gate.get(key)!r}, чекпойнт "
                     f"{ck.get(key)!r}")
    return p


def check_report(report, *, ck_sha, refine_sha, policy_sha, preflight=False):
    """Отчёт проверки вывода снят для этого чекпойнта этим кодом."""
    p = []
    if report is None:
        return ["нет отчёта проверки вывода"]

    def need(cond, msg):
        if not cond:
            p.append(msg)

    need(report.get("kind") == REPORT_KIND, f"отчёт: kind "
         f"{report.get('kind')!r}")
    need((report.get("verdict") or {}).get("passed") is True,
         "проверка вывода не пройдена")
    need(report.get("checkpoint_sha1") == ck_sha,
         "отчёт снят для другого файла чекпойнта")
    need(report.get("refine_module_sha1") == refine_sha,
         "отчёт снят другой версией k15d_depth_refine.py")
    need(report.get("policy_module_sha1") == policy_sha,
         "отчёт снят другой версией k15d_policy.py")
    if preflight:
        need(report.get("smoke") is True,
             "предполётная проверка — только со smoke-отчётом")
    else:
        need(report.get("smoke") is not True, "отчёт снят на smoke-чекпойнте")
    return p


class LevelLog:
    """Исполняемые шаги a0 и уровней по вызовам; take() отдаёт и обнуляет."""

    def __init__(self, names, h_exec):
        self.names = list(names)
        self.h_exec = int(h_exec)
        self.reset()

    def reset(self):
        self.a0, self.levels = [], {n: [] for n in self.names}

    def add(self, a0, levels):
        self.a0.append(np.asarray(a0[:, :self.h_exec], np.float32))
        for n in self.names:
            self.levels[n].append(
                np.asarray(levels[n][:, :self.h_exec], np.float32))

    def take(self):
        """(сводка для JSON, массивы для npz)."""
        if not self.a0:
            self.reset()
            return dict(calls=0), {}
        a0 = np.stack(self.a0)                          # [C, B, H, 7]
        arrays = {"k15d_a0": a0}
        summ = dict(calls=int(a0.shape[0]), levels=self.names)
        prev = a0
        for n in self.names:
            a = np.stack(self.levels[n])
            arrays[f"k15d_level_{n}"] = a
            summ[f"mean_abs_delta_{n}"] = np.abs(a - prev).reshape(
                -1, 7).mean(0).tolist()
            summ[f"absmax_{n}"] = np.abs(a).reshape(-1, 7).max(0).tolist()
            summ[f"finite_{n}"] = bool(np.isfinite(a).all())
            prev = a
        self.reset()
        return summ, arrays


def context_namespace(device, init_gate, root="third_party/actioncodec"):
    import k15_context
    ns = dict(k15_context.DEFAULTS)
    ns.update(device=str(device), root=root, init_gate=init_gate)
    return argparse.Namespace(**ns)


def load_refiner(ctx, checkpoint, *, k15d_gate, preflight=False):
    """Контекст, чекпойнт и гейт K-15d → установленное уточнение.

    Общая часть руки и проверки вывода: обе обязаны собрать одно и то же.
    """
    import k15d_check_init_identity as kg
    import k15d_depth_refine as dr
    import k15_train_depth_rvq as k15t
    from k15_train_depth_rvq import H_EXEC
    torch, model, dev = ctx.torch, ctx.model, ctx.dev
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    frozen, _n, _e = k15t.frozen_content_sha(model, torch, set())
    stats = dr.compute_stats(ctx, H_EXEC)
    gate = kg.check_gate_report(k15d_gate, expect=dict(
        device=str(dev), code=kg.code_shas(),
        architecture_code_version=ctx.code_version,
        joint_sha1=ctx.joint_sha, plan_sha1=ctx.q0_prov["plan_sha1"],
        frozen_sha1=frozen, stats_sha1=stats["sha1"], hp=dict(dr.HP)))
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    problems = check_checkpoint(ck, preflight=preflight)
    problems += check_gate_vs_ckpt(gate, ck, device=dev)
    if problems:
        raise SystemExit(f"чекпойнт {checkpoint} не годится для руки: "
                         + "; ".join(problems))
    ref = dr.DepthRefiner.from_export(
        ck["refiner"], norm_src=model.action_expert.norm,
        layers=model.action_expert.layers, device=dev)
    ref.attach(model)
    level = ARM_PHASES[ck["phase"]]
    train_names = {n for n, _ in ref.phase_named_parameters(ck["phase"])}
    if dr.state_sha(ref, train_names) != ck.get("selected_level_sha1"):
        raise SystemExit("уточнение загрузилось не той точкой")
    return SimpleNamespace(ref=ref, ck=ck, gate=gate, stats=stats,
                           level=level, frozen=frozen, h_exec=int(H_EXEC))


def make_act(ctx, L, *, log=None, verbose=True):
    """act(batch, pos_off, autocast, first) -> (чанк [B,T,7], коды q0)."""
    torch, model = ctx.torch, ctx.model
    ref, level = L.ref, L.level
    names = ref.level_names()

    def act(batch, pos_off, autocast, first):
        with torch.no_grad(), autocast:
            v, p_ = model.build_inputs(position_offset=pos_off, **batch)
            out = ref.run(model, vlm_inputs_embeds=v,
                          attention_mask=batch.get("attention_mask"),
                          position_ids=p_, decode=ctx.decode_fp32)
        a = out["actions"][level]
        if not bool(torch.isfinite(a).all()):
            raise SystemExit("рука k15d: действие не конечно")
        amax = float(a[:, :L.h_exec].abs().max())
        if amax > ACTION_CLIP_BOUND:
            raise SystemExit(f"рука k15d: |действие| {amax:.3f} выше предела "
                             f"{ACTION_CLIP_BOUND} в нормированной шкале")
        if first:
            moved = float((a - out["a0"]).abs().max())
            if moved <= 0.0:
                raise SystemExit("рука k15d: итоговый уровень равен a0 — "
                                 "веса уточнения не встали")
            if verbose:
                print(f"    проверка k15d: уровни {names}, исполняется "
                      f"{level}; max|a-a0| {moved:.4f}, max|a| {amax:.3f}",
                      flush=True)
        if log is not None:
            log.add(out["a0"].cpu().numpy(),
                    {n: out["actions"][n].cpu().numpy() for n in names})
        return a.float().cpu().numpy(), out["q0"].cpu().numpy()

    return act


def build_arm(device, checkpoint, inference_report, torch, *,
              preflight=False, init_gate=None, k15d_gate=None):
    """Рука целиком. Модель создаётся здесь — после сред харнесса."""
    import k15_context
    import k15d_depth_refine as dr
    g15a, g15d = gate_paths(device)
    init_gate, k15d_gate = init_gate or g15a, k15d_gate or g15d
    for f in (checkpoint, inference_report, init_gate, k15d_gate):
        if not f or not os.path.exists(f):
            raise SystemExit(f"нет {f}")
    with open(inference_report, encoding="utf-8") as fh:
        report = json.load(fh)
    ck_sha = sha_file(checkpoint)
    refine_sha = sha_file(dr.__file__)
    policy_sha = sha_file(os.path.abspath(__file__))
    problems = check_report(report, ck_sha=ck_sha, refine_sha=refine_sha,
                            policy_sha=policy_sha, preflight=preflight)
    if problems:
        raise SystemExit("рука K-15d не собрана: " + "; ".join(problems))
    ctx = k15_context.build(context_namespace(device, init_gate))
    L = load_refiner(ctx, checkpoint, k15d_gate=k15d_gate,
                     preflight=preflight)
    log = LevelLog(L.ref.level_names(), L.h_exec)
    act = make_act(ctx, L, log=log)
    ck = L.ck
    meta = dict(
        arm="k15d", phase=ck["phase"], variant=ck["variant"],
        level=L.level, levels=L.ref.level_names(),
        checkpoint=os.path.abspath(checkpoint), checkpoint_sha1=ck_sha,
        checkpoint_run_id=ck.get("run_id"),
        selected_tag=ck.get("selected_tag"),
        selected_level_sha1=ck.get("selected_level_sha1"),
        refine_module_sha1=refine_sha, policy_module_sha1=policy_sha,
        inference_report=os.path.abspath(inference_report),
        inference_report_sha1=sha_file(inference_report),
        k15d_gate_run_id=L.gate.get("run_id"),
        trained_gate_run_id=ck.get("gate_run_id"),
        stats_sha1=L.stats["sha1"], frozen_content_sha=L.frozen,
        k15a_gate=ctx.gate_info, code_version=ctx.code_version,
        joint_sha1=ctx.joint_sha, codec=ctx.codec_fp,
        admission=(ck.get("admission") or {}).get("admissible"),
        preflight=bool(preflight), decodes_per_call=1, layers_per_call=24)
    import k15_train_depth_rvq as k15t
    try:
        meta = json.loads(json.dumps(meta, default=k15t.json_scalar))
    except TypeError as e:
        raise SystemExit(f"метаданные руки не сериализуются: {e}")
    meta["model_fingerprint"] = hashlib.sha1("|".join(
        str(meta[k]) for k in ("checkpoint_sha1", "refine_module_sha1",
                               "policy_module_sha1", "selected_level_sha1",
                               "frozen_content_sha", "joint_sha1")
    ).encode()).hexdigest()[:12]
    return SimpleNamespace(model=ctx.model, proc=ctx.proc, codec=ctx.codec,
                           act=act, log=log, meta=meta, ctx=ctx)


def selftest():
    good = dict(kind="k15d_phase", phase="d0", status="complete", final=True,
                technical_ok=True, selected_tag="step005000", mode="full",
                admission=dict(admissible=True), stats_sha1="S",
                frozen_sha1="F")
    assert check_checkpoint(good) == []
    for why, mut in (("фаза h1p1", dict(phase="h1p1")),
                     ("partial", dict(status="partial", final=False)),
                     ("validating", dict(status="validating")),
                     ("технический отказ", dict(technical_ok=False)),
                     ("нет точки", dict(selected_tag=None)),
                     ("smoke", dict(mode="smoke")),
                     ("не допущен", dict(admission=dict(admissible=False)))):
        assert check_checkpoint(dict(good, **mut)), why
    sm = dict(good, mode="smoke", admission=dict(admissible=False))
    assert check_checkpoint(sm, preflight=True) == []
    assert check_checkpoint(good, preflight=True)       # настоящий — нельзя
    assert check_checkpoint(dict(good, phase="h1p2")) == []

    gate = dict(device="cuda:0", stats_sha1="S", frozen_sha1="F")
    assert check_gate_vs_ckpt(gate, good, device="cuda:0") == []
    assert check_gate_vs_ckpt(gate, good, device="cuda:1")
    assert check_gate_vs_ckpt(dict(gate, stats_sha1="X"), good,
                              device="cuda:0")

    rep = dict(kind=REPORT_KIND, verdict=dict(passed=True),
               checkpoint_sha1="C", refine_module_sha1="R",
               policy_module_sha1="P", smoke=False)
    kw = dict(ck_sha="C", refine_sha="R", policy_sha="P")
    assert check_report(rep, **kw) == []
    for why, mut in (("не пройден", dict(verdict=dict(passed=False))),
                     ("другой файл", dict(checkpoint_sha1="X")),
                     ("другой модуль уточнения", dict(refine_module_sha1="X")),
                     ("другой модуль руки", dict(policy_module_sha1="X")),
                     ("smoke-отчёт", dict(smoke=True)),
                     ("другой kind", dict(kind="x"))):
        assert check_report(dict(rep, **mut), **kw), why
    assert check_report(None, **kw)
    assert check_report(dict(rep, smoke=True), preflight=True, **kw) == []
    assert check_report(rep, preflight=True, **kw)

    assert gate_paths("cuda:1") == ("reports/k15a/init_identity.json",
                                    "reports/k15d/init_identity.json")
    assert gate_paths("cuda:0")[1] == "reports/k15d/init_identity_cuda0.json"

    lg = LevelLog(["1", "2"], 8)
    a0 = np.zeros((5, 20, 7), np.float32)
    lv = {"1": a0 + 0.1, "2": a0 + 0.3}
    lg.add(a0, lv)
    lg.add(a0, lv)
    s, arr = lg.take()
    assert s["calls"] == 2 and arr["k15d_a0"].shape == (2, 5, 8, 7)
    assert abs(s["mean_abs_delta_1"][0] - 0.1) < 1e-6
    assert abs(s["mean_abs_delta_2"][0] - 0.2) < 1e-6
    assert lg.take()[0]["calls"] == 0
    print("самопроверка k15d_policy пройдена")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="K-15d: рука уточнения")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    if a.selftest:
        sys.exit(selftest())
    raise SystemExit("это модуль руки для k9h_multiarm_gate --policy k15d")
