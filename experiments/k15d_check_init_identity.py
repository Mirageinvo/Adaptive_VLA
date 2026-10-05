#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15d: гейт нулевой точки уточнения D0/H1 на настоящей модели.

Архитектура новая, K-15a её не аттестует. Гейт короткий (план §5) и
проверяет на нескольких батчах val_sel:

  1. три пути q0 совпадают побитово: канонический план, проход
     `forward_depth_aligned_rvq(mode="fast")` и префикс K-15d;
  2. a0 из прохода совпадает (допуск 1e-5) с a0 из декодирования
     канонических кодов — на этом держится статистика train, посчитанная
     без прохода VLM;
  3. в нулевой точке a_d = a1 = a2 = a0 побитово, все значения конечны;
  4. белые списки фаз точны, веса модели не обучаются;
  5. причинность после искусственной ненулевой инициализации: голова и
     LoRA меняют выход; φ меняет выход (обнуление и перемешивание входа),
     а при нулевом φ перемешивание НЕ меняет — контроль вырожденности;
     у H1 обнуление φ2 меняет a2 и не трогает a1;
  6. градиент фазы доходит до каждого тензора белого списка и только до
     него;
  7. сохранение и загрузка дают побитово тот же выход;
  8. отпечаток замороженного до и после совпадает.

Артефакт `reports/k15d/init_identity.json` требуют тренер и проверка
вывода: отпечатки кода уточнения, статистики и замороженного.
"""
import argparse
import datetime
import json
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k15_context  # noqa: E402
import k15d_depth_refine as dr  # noqa: E402
from k15_train_depth_rvq import H_EXEC  # noqa: E402

GATE_KIND = "k15d_init_identity"


def strided(n, k):
    if k >= n:
        return list(range(n))
    return sorted({int(round(x)) for x in np.linspace(0, n - 1, k)})


def code_shas():
    return {f: k15_context.sha12(os.path.join(HERE, f))
            for f in ("k15d_depth_refine.py", "k15d_check_init_identity.py")}


def check_gate_report(path, *, expect):
    """Тренер и проверка вывода: гейт снят в этой же обстановке."""
    if not os.path.exists(path):
        raise SystemExit(f"нет гейта K-15d {path}: сначала "
                         f"k15d_check_init_identity.py")
    g = json.load(open(path))
    if g.get("kind") != GATE_KIND or g.get("passed") is not True:
        raise SystemExit(f"{path}: kind {g.get('kind')!r}, passed "
                         f"{g.get('passed')!r}")
    bad = [f"{k}: гейт {g.get(k)!r}, сейчас {v!r}"
           for k, v in expect.items() if g.get(k) != v]
    if bad:
        raise SystemExit("гейт K-15d снят в другой обстановке: "
                         + "; ".join(bad))
    return g


def main():
    ap = argparse.ArgumentParser(description="K-15d: гейт нулевой точки")
    k15_context.add_common_arguments(ap)
    ap.add_argument("--out", default="reports/k15d/init_identity.json")
    ap.add_argument("--gate-batches", type=int, default=4)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    t0 = time.time()
    ctx = k15_context.build(a)
    torch, model, dev = ctx.torch, ctx.model, ctx.dev
    k15t = k15_context.k15t
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    frozen0, n_frozen, n_elem = k15t.frozen_content_sha(model, torch, set())
    print(f"  замороженное: {n_frozen} тензоров, {n_elem} значений, "
          f"отпечаток {frozen0}")
    stats = dr.compute_stats(ctx, H_EXEC)
    print(f"  статистика train ({stats['rows']} строк, {stats['sha1']}): "
          f"граница руки " + ", ".join(f"{x:.3f}" for x in
                                       stats["delta_scale"][:6])
          + "; RMS остатка " + ", ".join(f"{x:.3f}" for x in
                                          stats["loss_scale"]))
    val = list(ctx.parts_full["val_sel"])
    batches = [val[i] for i in strided(len(val), int(a.gate_batches))]
    q0_dev = torch.as_tensor(np.asarray(ctx.q0_can), device=dev)
    ac16 = torch.autocast(device_type=dev.type, dtype=ctx.dt)
    checks = {}

    def ok(name, cond, detail=""):
        """Повтор имени (по батчам) сводится по И, а не перезаписывается:
        иначе отказ на первом батче скрыл бы успех последнего."""
        prev = checks.get(name)
        n = 1 if prev is None else prev["n"] + 1
        passed = bool(cond) and (prev is None or prev["passed"])
        checks[name] = dict(passed=passed, n=n, detail=str(detail))
        if prev is None or not cond:
            print(f"  [{'OK' if cond else 'ОТКАЗ'}] {name} {detail}")
        return bool(cond)

    inputs = []
    with torch.no_grad():
        for po, sel in batches:
            b = ctx.build_batch(po, sel)
            with ac16:
                v, p_ids = model.build_inputs(position_offset=po, **b)
                ref_out = model.forward_depth_aligned_rvq(
                    vlm_inputs_embeds=v,
                    attention_mask=b.get("attention_mask"),
                    position_ids=p_ids, mode="fast", tau=1.0)
            inputs.append(dict(sel=sel, v=v, p_ids=p_ids,
                               am=b.get("attention_mask"),
                               q_ref=ref_out["pred_codes"][0]))

    def run(ref, x, **kw):
        with ac16:
            return ref.run(model, vlm_inputs_embeds=x["v"],
                           attention_mask=x["am"], position_ids=x["p_ids"],
                           decode=ctx.decode_fp32, **kw)

    variants = {}
    for variant in dr.VARIANTS:
        ref, n_hooks = dr.make_refiner(ctx, variant, stats, a.seed)
        names = ref.level_names()
        phases = [p for p, v in dr.PHASE_VARIANT.items() if v == variant]
        tag = variant
        init_obj = ref.export()
        print(f"  {variant}: хуков LoRA {n_hooks}, параметров "
              f"{ref.count() / 1e6:.3f} млн (" + ", ".join(
                  f"{p} {ref.count(p) / 1e6:.3f}" for p in phases) + ")")
        # 1–3. нулевая точка
        all_ok = True
        with torch.no_grad():
            for x in inputs:
                o = run(ref, x)
                sel_t = torch.as_tensor(x["sel"], device=dev)
                a0_codes = dr.decode_q0_rows(
                    ctx, np.asarray(x["sel"], np.int64)).to(dev)
                all_ok &= ok(f"{tag}.q0_three_paths",
                             torch.equal(o["q0"], x["q_ref"])
                             and torch.equal(o["q0"], q0_dev[sel_t]))
                # Допуск, а не биты: статистика декодирует кусками по 2048
                # строк, а проход — по 8, и ядра matmul на GPU могут
                # различаться в последних разрядах. Статистике нужна
                # близость, q0 при этом сверен побитово выше.
                gap = float((o["a0"] - a0_codes).abs().max())
                all_ok &= ok(f"{tag}.a0_pass_matches_codes", gap <= 1e-5,
                             f"макс. расхождение {gap:.2e}")
                for n in names:
                    all_ok &= ok(f"{tag}.identity_{n}",
                                 torch.equal(o["actions"][n], o["a0"]))
                    all_ok &= ok(f"{tag}.finite_{n}", bool(
                        torch.isfinite(o["actions"][n]).all()))
        # хуки LoRA: каждый сработал ровно один раз за проход
        ref.fire_counts.clear()
        with torch.no_grad():
            run(ref, inputs[0])
        exp = ref.expected_hooks()
        got = ref.fire_counts
        ok(f"{tag}.lora_hooks_fire_once",
           set(got) == exp and set(got.values()) == {1},
           f"{len(got)}/{len(exp)} хуков, срабатываний "
           f"{sorted(set(got.values()))}")
        # 4. белые списки
        want_pref = {p: (f"feedback.{dr.PHASES[p]}.",
                         f"heads.{dr.PHASES[p]}.", f"lora.{dr.PHASES[p]}.")
                     for p in phases}
        for p in phases:
            got = set(ref.set_phase(p))
            want = {n for n, _ in ref.named_parameters()
                    if n.startswith(want_pref[p])}
            ok(f"{tag}.whitelist_{p}", got == want and len(got) > 0,
               f"{len(got)} тензоров")
        ref.set_phase(None)
        ok(f"{tag}.model_frozen",
           not any(p.requires_grad for p in model.parameters()))
        # 5. причинность после ненулевой инициализации
        x = inputs[0]
        dr._perturb(ref, ("heads.",), scale=0.02, seed=21)
        with torch.no_grad():
            o_head = run(ref, x)
        dr._perturb(ref, ("lora.",), scale=0.02, seed=23)
        with torch.no_grad():
            o_h = run(ref, x)
            for n in names:
                ok(f"{tag}.lora_changes_output_{n}",
                   not torch.equal(o_h["actions"][n], o_head["actions"][n]))
            for n in names:
                ok(f"{tag}.head_lora_change_{n}",
                   not torch.equal(o_h["actions"][n], o_h["a0"]))
                o_s = run(ref, x, fb_mode={n: "shuffle"})
                ok(f"{tag}.control_zero_phi_shuffle_inert_{n}",
                   torch.equal(o_s["actions"][n], o_h["actions"][n]))
            dr._perturb(ref, ("feedback.",), scale=0.02, seed=22)
            o_f = run(ref, x)
            for n in names:
                o_s = run(ref, x, fb_mode={n: "shuffle"})
                o_z = run(ref, x, fb_mode={n: "zero"})
                d_s = float((o_s["actions"][n] - o_f["actions"][n]).abs()
                            .max())
                d_z = float((o_z["actions"][n] - o_f["actions"][n]).abs()
                            .max())
                ok(f"{tag}.phi_causal_{n}", d_s > 0 and d_z > 0,
                   f"перемешивание {d_s:.2e}, обнуление {d_z:.2e}")
            if variant == "h1":
                o_z2 = run(ref, x, fb_mode={"2": "zero"})
                ok(f"{tag}.phi2_leaves_a1",
                   torch.equal(o_z2["actions"]["1"], o_f["actions"]["1"]))
        # 6. градиенты фаз
        for p in phases:
            lv = dr.PHASES[p]
            train = set(ref.set_phase(p))
            act = torch.from_numpy(np.asarray(
                ctx.ACT[x["sel"]], np.float32)).to(dev)
            with ac16:
                o = ref.run(model, vlm_inputs_embeds=x["v"],
                            attention_mask=x["am"], position_ids=x["p_ids"],
                            decode=ctx.decode_fp32, train_level=lv)
            loss, _ = dr.level_loss(o["actions"][lv], o["logits"][lv],
                                    act[:, :o["a0"].shape[1], :7],
                                    ref.loss_scale, H_EXEC)
            ok(f"{tag}.loss_finite_{p}", bool(torch.isfinite(loss)),
               f"{float(loss):.4e}")
            (loss * 1024.0).backward()
            # Живой градиент — конечный и ненулевой: NaN и Inf прошли бы
            # проверку «сумма не равна нулю».
            dead = [n for n, q in ref.named_parameters() if n in train
                    and (q.grad is None
                         or not bool(torch.isfinite(q.grad).all())
                         or float(q.grad.abs().sum()) == 0.0)]
            leak = [n for n, q in ref.named_parameters() if n not in train
                    and q.grad is not None]
            mleak = sum(1 for q in model.parameters() if q.grad is not None)
            ok(f"{tag}.grad_{p}", not dead and not leak and not mleak,
               f"без конечного ненулевого градиента {dead[:3]}, утечка {leak[:3]}, модель "
               f"{mleak}")
            ref.zero_grad(set_to_none=True)
        ref.set_phase(None)
        # 7. сохранение и загрузка
        with torch.no_grad():
            o_before = run(ref, x)
        obj = ref.export()
        with tempfile.TemporaryDirectory() as td:
            fn = os.path.join(td, "ref.pt")
            torch.save(obj, fn)
            obj2 = torch.load(fn, map_location="cpu", weights_only=False)
        ref.detach_hooks()
        torch.manual_seed(int(a.seed) + 777)
        ref2 = dr.DepthRefiner.from_export(
            obj2, norm_src=model.action_expert.norm,
            layers=model.action_expert.layers, device=dev)
        ref2.attach(model)
        with torch.no_grad():
            o_after = run(ref2, x)
        ok(f"{tag}.save_load", all(
            torch.equal(o_before["actions"][n], o_after["actions"][n])
            for n in names))
        ref2.detach_hooks()
        # возврат в нулевую точку из экспорта начального состояния
        ref3 = dr.DepthRefiner.from_export(
            init_obj, norm_src=model.action_expert.norm,
            layers=model.action_expert.layers, device=dev)
        ref3.attach(model)
        with torch.no_grad():
            o3 = run(ref3, x)
        ok(f"{tag}.restore_identity", all(
            torch.equal(o3["actions"][n], o3["a0"]) for n in names))
        ref3.detach_hooks()
        variants[variant] = dict(
            n_hooks=n_hooks, params=ref.count(),
            phase_params={p: ref.count(p) for p in phases},
            init_state_sha1=init_obj["state_sha1"], all_ok=bool(all_ok))
        del ref, ref2, ref3

    frozen1, _n, _e = k15t.frozen_content_sha(model, torch, set())
    ok("frozen_unchanged", frozen1 == frozen0, f"{frozen0} -> {frozen1}")
    passed = all(c["passed"] for c in checks.values())
    rep = dict(
        kind=GATE_KIND, passed=bool(passed),
        run_id=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        device=str(dev), compute_dtype=a.dtype, seed=int(a.seed),
        code=code_shas(), architecture_code_version=ctx.code_version,
        k15a_gate=dict(ctx.gate_info), joint_sha1=ctx.joint_sha,
        plan_sha1=ctx.q0_prov["plan_sha1"], codec=ctx.codec_fp,
        frozen_sha1=frozen0, frozen_tensors=n_frozen,
        stats=stats, stats_sha1=stats["sha1"], hp=dict(dr.HP),
        h_exec=int(H_EXEC), git_head=ctx.git_head, dirty=bool(ctx.dirty),
        gate_batches=[[int(po), [int(r) for r in sel]]
                      for po, sel in batches],
        variants=variants, checks=checks,
        seconds=round(time.time() - t0, 1))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False,
                  default=k15t.json_scalar)
    os.replace(tmp, a.out)
    n_fail = sum(1 for c in checks.values() if not c["passed"])
    print(f"ГЕЙТ K-15d: {'ПРОЙДЕН' if passed else 'НЕ ПРОЙДЕН'} "
          f"({len(checks) - n_fail}/{len(checks)} проверок), {a.out}")
    return 0 if passed else 3


def selftest():
    """Без GPU: сверка отчёта гейта отказывает на любом расхождении."""
    with tempfile.TemporaryDirectory() as td:
        fn = os.path.join(td, "g.json")
        json.dump(dict(kind=GATE_KIND, passed=True, device="cuda:1",
                       stats_sha1="abc"), open(fn, "w"))
        check_gate_report(fn, expect=dict(device="cuda:1",
                                          stats_sha1="abc"))
        for bad in (dict(device="cuda:0"), dict(stats_sha1="x"),
                    dict(missing_key="v")):
            try:
                check_gate_report(fn, expect=dict(
                    dict(device="cuda:1", stats_sha1="abc"), **bad))
                raise AssertionError(f"прошло расхождение {bad}")
            except SystemExit:
                pass
        json.dump(dict(kind=GATE_KIND, passed=False), open(fn, "w"))
        try:
            check_gate_report(fn, expect={})
            raise AssertionError("прошёл непройденный гейт")
        except SystemExit:
            pass
    assert strided(10, 4) == [0, 3, 6, 9] and strided(3, 5) == [0, 1, 2]
    assert set(code_shas()) == {"k15d_depth_refine.py",
                                "k15d_check_init_identity.py"}
    print("самопроверка k15d_check_init_identity пройдена")
    return 0


if __name__ == "__main__":
    sys.exit(main())
