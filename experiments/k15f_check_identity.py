#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15f: гейт тождества и механики базиса на настоящей модели.

На нескольких батчах val_sel (те же батчи плана, что в кэше):

  1. три пути q0 совпадают побитово: канонический, forward_depth_aligned_rvq
     (mode="fast") и чистый проход `plain_h18`;
  2. h18 чистого прохода побитово равен кэшу, на котором учился базис;
  3. a0 прохода совпадает с декодом канонических кодов (допуск 1e-5);
  4. c = 0 даёт действие, побитово равное a0 (и learned, и контроль);
  5. каждый ±e_j меняет действие; контроль при том же c отличается от
     learned и сохраняет нормы направлений;
  6. h18 влияет на базис (если выбранная точка не чистая PCA — иначе это
     фиксируется как «базис не зависит от состояния»);
  7. |a| <= 1.5 при ±e_j learned и контроля с выбранным множителем;
  8. градиент потери базиса конечен и ненулев у каждого тензора головы и
     не доходит до модели;
  9. сохранение и загрузка базиса воспроизводят базис побитово;
 10. отпечаток замороженного до и после совпадает.

Отчёт reports/k15f/identity_<карта>.json требует рука k15f.
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
import k15f_build_cache as kb  # noqa: E402
import k15f_continuous_refine as kf  # noqa: E402

KIND = "k15f_identity"
ACTION_CLIP_BOUND = 1.5


def load_head(torch, ck, model, dev):
    """Голова базиса из чекпойнта с нормой, сверенной с моделью."""
    w, eps = kf.norm_of(model)
    if abs(eps - float(ck["norm_eps"])) > 0 or not torch.equal(
            w, ck["state"]["norm.weight"].float().cpu()):
        raise SystemExit("финальная норма модели не та, что в кэше базиса")
    head = kf.BasisHead(w, eps, ck["d_model"], ck["n_pos"], ck["stats"],
                        hp=ck["hp"]).to(dev)
    miss, unexp = head.load_state_dict(
        {k: v.to(dev) for k, v in ck["state"].items()}, strict=True)
    if miss or unexp:
        raise SystemExit(f"состояние головы: нет {miss}, лишние {unexp}")
    if kf.state_sha(head) != ck["state_sha1"]:
        raise SystemExit("отпечаток головы базиса не совпал")
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return head


def check_basis_ckpt(ck):
    p = []
    if ck.get("kind") != "k15f_basis" or ck.get("status") != "complete":
        p.append(f"kind/status {ck.get('kind')}/{ck.get('status')}")
    if ck.get("technical_ok") is not True:
        p.append("базис технически не исправен")
    if ck.get("mode") != "full":
        p.append(f"режим базиса {ck.get('mode')!r}, нужен full")
    if ck.get("amp_factor") is None:
        p.append("множитель амплитуды не выбран")
    return p


def main():
    ap = argparse.ArgumentParser(description="K-15f: гейт тождества")
    k15_context.add_common_arguments(ap)
    ap.add_argument("--basis", default="data/k15f/basis_s0.pt")
    # не --cache: этот флаг уже занят общими аргументами k15_context (кэш
    # K-11a), argparse отказал бы на конфликте
    ap.add_argument("--h18-cache", default="data/k15f/h18_cache")
    ap.add_argument("--gate-batches", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.time()
    import torch
    ck = torch.load(a.basis, map_location="cpu", weights_only=False)
    prob = check_basis_ckpt(ck)
    if prob:
        raise SystemExit("базис не годится: " + "; ".join(prob))
    C = kb.load_cache(a.h18_cache)
    if kb.sha_file(os.path.join(a.h18_cache, "manifest.json")) != \
            ck["cache_manifest_sha1"]:
        raise SystemExit("базис обучен на другом кэше")
    ctx = k15_context.build(a)
    torch, model, dev = ctx.torch, ctx.model, ctx.dev
    k15t = k15_context.k15t
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    frozen0, _n, _e = k15t.frozen_content_sha(model, torch, set())
    env = kb.check_env(C["manifest"], ctx, frozen0)
    if env:
        raise SystemExit("кэш снят в другой обстановке: " + "; ".join(env))
    head = load_head(torch, ck, model, dev)
    f = float(ck["amp_factor"])
    R = kf.rotation(int(ck["control_seed"])).to(dev)
    checks = {}

    def ok(name, cond, detail=""):
        prev = checks.get(name)
        passed = bool(cond) and (prev is None or prev["passed"])
        checks[name] = dict(passed=passed, detail=str(detail))
        if prev is None or not cond:
            print(f"  [{'OK' if cond else 'ОТКАЗ'}] {name} {detail}")

    val = list(ctx.parts_full["val_sel"])
    batches = [val[i] for i in kb.strided(len(val), int(a.gate_batches))]
    slot = {int(r): i for i, r in enumerate(C["val_sel"]["rows"])}
    q0_dev = torch.as_tensor(np.asarray(ctx.q0_can), device=dev)
    ac16 = torch.autocast(device_type=dev.type, dtype=ctx.dt)
    rels = []
    for bi, (po, sel) in enumerate(batches):
        b = ctx.build_batch(po, sel)
        with torch.no_grad(), ac16:
            v, pid = model.build_inputs(position_offset=po, **b)
            ref_out = model.forward_depth_aligned_rvq(
                vlm_inputs_embeds=v, attention_mask=b.get("attention_mask"),
                position_ids=pid, mode="fast", tau=1.0)
            q0, a0, h18 = kf.plain_h18(
                model, vlm_inputs_embeds=v,
                attention_mask=b.get("attention_mask"), position_ids=pid,
                decode=ctx.decode_fp32)
        sel_t = torch.as_tensor(sel, device=dev)
        ok("q0_three_paths", torch.equal(q0, ref_out["pred_codes"][0])
           and torch.equal(q0, q0_dev[sel_t]))
        idx = [slot.get(int(r)) for r in sel]
        # ОТСУТСТВИЕ СТРОКИ — ОТКАЗ, а не пропуск проверки
        ok("rows_in_cache", all(i is not None for i in idx),
           f"нет {sum(i is None for i in idx)} строк")
        if all(i is not None for i in idx):
            cached = torch.from_numpy(np.asarray(
                C["val_sel"]["h18"][np.array(idx)])).to(dev)
            # в кэше строка из ПЕРВОГО батча плана; гейт идёт по тем же
            # батчам, поэтому совпадение побитовое
            ok("h18_equals_cache", torch.equal(h18.half(), cached))
        a0c = dr.decode_q0_rows(ctx, np.asarray(sel, np.int64))
        gap = float((a0.cpu() - a0c).abs().max())
        ok("a0_matches_codes", gap <= 1e-5, f"{gap:.2e}")
        with torch.no_grad():
            basis, _u = head(h18, a0)
            basis = basis * f
            for fam, bb in (("learned", basis), ("control", basis @ R)):
                a_z = kf.combine(a0, kf.compose(bb, torch.zeros(
                    kf.K_BASIS, device=dev)), head.sigma_arm, head.sigma_g)
                ok(f"zero_is_a0_{fam}", torch.equal(a_z, a0))
                for j in range(kf.K_BASIS):
                    for sgn in (1.0, -1.0):
                        c = torch.zeros(kf.K_BASIS, device=dev)
                        c[j] = sgn
                        a_c = kf.combine(a0, kf.compose(bb, c),
                                         head.sigma_arm, head.sigma_g)
                        ok(f"e_changes_{fam}", not torch.equal(a_c, a0))
                        ok(f"range_{fam}", bool(torch.isfinite(a_c).all())
                           and float(a_c[:, :kf.H_EXEC].abs().max())
                           <= ACTION_CLIP_BOUND,
                           f"max|a| {float(a_c[:, :kf.H_EXEC].abs().max()):.3f}")
            ok("control_differs", not torch.allclose(basis @ R, basis))
            ok("control_keeps_norms", torch.allclose(
                (basis @ R).norm(dim=-1), basis.norm(dim=-1), atol=1e-4))
            # ЗАВИСИМОСТЬ ОТ h18: относительный RMS изменения базиса при
            # перестановке h18 между строками (a0 — свой)
            b_roll, _ = head(h18.roll(1, 0), a0)
            rel = float((b_roll * f - basis).pow(2).mean().sqrt()
                        / basis.pow(2).mean().sqrt().clamp_min(1e-12))
            rels.append(rel)
            act_b = torch.from_numpy(np.asarray(ctx.ACT[sel], np.float32)
                                     [:, :a0.shape[1], :7]).to(dev)
            rr = kf.to_tangent(act_b, a0, head.sigma_arm, head.sigma_g
                               ).reshape(len(sel), -1)
            _bb, uu = head(h18, a0)
            geo = kf.basis_diagnostics(head, basis, uu, rr)
            ok("geometry", kf.geometry_ok(geo),
               f"обусловленность {geo['cond_max']:.2f}, попарный косинус "
               f"{geo['pair_cos_max']:.3f}")
    # градиенты головы (на копии, требующей градиента)
    head_g = load_head(torch, ck, model, dev)
    for p in head_g.parameters():
        p.requires_grad_(True)
    po, sel = batches[0]
    b = ctx.build_batch(po, sel)
    with torch.no_grad(), ac16:
        v, pid = model.build_inputs(position_offset=po, **b)
        _q, a0, h18 = kf.plain_h18(
            model, vlm_inputs_embeds=v,
            attention_mask=b.get("attention_mask"), position_ids=pid,
            decode=ctx.decode_fp32)
    act = torch.from_numpy(np.asarray(ctx.ACT[sel], np.float32)
                           [:, :a0.shape[1], :7]).to(dev)
    r = kf.to_tangent(act, a0, head_g.sigma_arm, head_g.sigma_g).reshape(
        len(sel), -1)
    # КОНТРОЛЬ ВЫРОЖДЕННОСТИ: при нулевом выходном слое (эпоха 0 — чистая
    # PCA) вход головы градиента не получает по построению; проверка
    # градиентов на такой голове была бы ложным отказом, а на ненулевой —
    # содержательной. Поэтому градиенты проверяются после искусственной
    # ненулевой инициализации выхода, как в гейте K-15d.
    with torch.no_grad():
        head_g.out.weight.normal_(0.0, 1e-3)
    bb, uu = head_g(h18, a0)
    loss, _parts = kf.basis_loss(head_g, bb, uu, r)
    loss.backward()
    dead = [n for n, p in head_g.named_parameters()
            if p.grad is None or not bool(torch.isfinite(p.grad).all())
            or float(p.grad.abs().sum()) == 0.0]
    leak = sum(1 for p in model.parameters() if p.grad is not None)
    ok("grads", not dead and not leak, f"мёртвые {dead[:3]}, утечка {leak}")
    # сохранение и загрузка
    with tempfile.TemporaryDirectory() as td:
        fn = os.path.join(td, "b.pt")
        torch.save(ck, fn)
        ck2 = torch.load(fn, map_location="cpu", weights_only=False)
    head2 = load_head(torch, ck2, model, dev)
    with torch.no_grad():
        b1, _ = head(h18, a0)
        b2, _ = head2(h18, a0)
    ok("save_load", torch.equal(b1, b2))
    frozen1, _n, _e = k15t.frozen_content_sha(model, torch, set())
    ok("frozen_unchanged", frozen1 == frozen0)
    passed = all(c["passed"] for c in checks.values())
    rel_med = float(np.median(rels)) if rels else 0.0
    state_dep = rel_med >= kf.STATE_DEP_MIN_REL
    # ИСХОД: 0 — иерархический базис (зависит от h18); 4 — технически
    # исправен, но фактически глобальная PCA: отдельный baseline, M1 для
    # иерархии не запускается; 3 — технический отказ
    code = 3 if not passed else (0 if state_dep else 4)
    print(f"  базис зависит от h18: {state_dep} — относительный RMS "
          f"изменения при перестановке h18 {rel_med:.4f} (порог "
          f"{kf.STATE_DEP_MIN_REL}); выбрана точка {ck.get('selected')}")
    tag = str(dev).replace(":", "")
    out = a.out or f"reports/k15f/identity_{tag}.json"
    rep = dict(kind=KIND, passed=bool(passed), code=code, checks=checks,
               state_dep_rel_rms=rel_med,
               state_dep_threshold=kf.STATE_DEP_MIN_REL,
               basis=os.path.abspath(a.basis),
               basis_sha1=kb.sha_file(a.basis),
               basis_state_sha1=ck["state_sha1"],
               basis_selected=ck.get("selected"),
               basis_state_dependent=bool(state_dep),
               amp_factor=f, control_seed=int(ck["control_seed"]),
               device=str(dev), compute_dtype=a.dtype,
               code_sha=dict(k15f_continuous_refine=kb.sha_file(kf.__file__),
                         k15f_check_identity=kb.sha_file(
                             os.path.abspath(__file__))),
               frozen_sha1=frozen0, joint_sha1=ctx.joint_sha,
               code_version=ctx.code_version, git_head=ctx.git_head,
               run_id=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
               seconds=round(time.time() - t0, 1))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out + ".tmp", "w") as fh:
        json.dump(rep, fh, indent=1, ensure_ascii=False,
                  default=k15t.json_scalar)
    os.replace(out + ".tmp", out)
    verdict = {0: "ПРОЙДЕН: базис зависит от h18",
               4: "технически исправен, но базис — ГЛОБАЛЬНАЯ PCA (код 4): "
                  "не иерархия, M1 для K-15f не запускается",
               3: "НЕ ПРОЙДЕН"}[code]
    print(f"ГЕЙТ K-15f: {verdict}; {out}")
    return code




def C_norm(ck):
    """(вес, eps) нормы из состояния чекпойнта базиса."""
    return ck["state"]["norm.weight"].float(), float(ck["norm_eps"])


def integration():
    """Кэш -> предобучение -> гейт -> рука на игрушечной среде K-15d (CPU)."""
    import tempfile
    import torch
    import k15d_train as kt
    import k15f_build_cache as kbc
    import k15f_policy as kpf
    import k15f_pretrain_basis as kpb
    import k15d_policy as kpd
    saved = (k15_context.build, sys.argv, kpd.gate_paths)
    ctx = kt._fake_ctx()
    ctx.model.action_expert.norm.variance_epsilon = 1e-6
    ctx.vocab = int(ctx.model.book0.shape[0])
    k15_context.build = lambda a: ctx
    try:
        with tempfile.TemporaryDirectory() as td:
            cache = os.path.join(td, "h18_cache")
            sys.argv = ["x", "--device", "cpu", "--out", cache]
            assert kbc.main() == 0
            man = json.load(open(os.path.join(cache, "manifest.json")))
            assert man["k15c_reuse_check"]["feedback_mask"] is None
            basis = os.path.join(td, "basis.pt")
            sys.argv = ["x", "--mode", "full", "--device", "cpu",
                        "--cache", cache, "--epochs", "1", "--out", basis,
                        "--report", os.path.join(td, "basis.json")]
            assert kpb.main() == 0
            ident = os.path.join(td, "identity.json")
            sys.argv = ["x", "--device", "cpu", "--basis", basis,
                        "--h18-cache", cache, "--out", ident,
                        "--gate-batches", "2"]
            code = main()
            assert code in (0, 4), f"гейт тождества: код {code}"
            # БАЗИС, ЗАВИСЯЩИЙ ОТ h18 (как после обучения): норма — буфер,
            # не меняется, и гейт обязан пройти с кодом 0
            ckb = torch.load(basis, map_location="cpu", weights_only=False)
            st_ = dict(ckb["state"])
            g_ = torch.Generator().manual_seed(1)
            st_["out.weight"] = torch.randn(st_["out.weight"].shape,
                                            generator=g_) * 0.05
            head_t = kf.BasisHead(*C_norm(ckb), ckb["d_model"], ckb["n_pos"],
                                  ckb["stats"], hp=ckb["hp"])
            head_t.load_state_dict(st_)
            basis_dep = os.path.join(td, "basis_dep.pt")
            torch.save(dict(ckb, state=st_, state_sha1=kf.state_sha(head_t),
                            selected="epoch1"), basis_dep)
            ident_dep = os.path.join(td, "identity_dep.json")
            sys.argv = ["x", "--device", "cpu", "--basis", basis_dep,
                        "--h18-cache", cache, "--out", ident_dep,
                        "--gate-batches", "2"]
            assert main() == 0, "гейт на зависящем от h18 базисе не прошёл"
            rd = json.load(open(ident_dep))
            assert rd["basis_state_dependent"] and rd["checks"][
                "frozen_unchanged"]["passed"]
            rep0 = json.load(open(ident))
            assert rep0["code"] == code
            if code == 4:
                # рука обязана отказать на базисе без зависимости от h18;
                # для проверки механики руки дальше — отчёт с кодом 0
                gate4 = os.path.join(td, "k15a4.json")
                open(gate4, "w").write("{}")
                kpd.gate_paths = lambda device: (gate4, gate4)
                try:
                    kpf.build_arm("cpu", basis, "z", torch,
                                  identity_report=ident)
                    raise AssertionError("рука приняла глобальную PCA")
                except SystemExit as e:
                    assert "h18" in str(e), e
                json.dump(dict(rep0, code=0, basis_state_dependent=True),
                          open(ident, "w"))
            gate = os.path.join(td, "k15a.json")
            open(gate, "w").write("{}")
            kpd.gate_paths = lambda device: (gate, gate)
            po, sel = ctx.parts_full["val_sel"][0]
            ac = torch.autocast(device_type="cpu", dtype=torch.bfloat16)
            acts = {}
            for lab in ("z", "l0p", "l0m", "r0p"):
                arm = kpf.build_arm("cpu", basis, lab, torch,
                                    identity_report=ident)
                a_np, q0c = arm.act(ctx.build_batch(po, sel), po, ac, True)
                assert np.array_equal(q0c, ctx.q0_can[np.asarray(sel)])
                acts[lab] = a_np
                assert arm.log.take()[0]["calls"] == 1
            assert not np.array_equal(acts["l0p"], acts["z"])
            assert not np.array_equal(acts["r0p"], acts["l0p"])
            # рука отвергает отчёт гейта для другого файла базиса
            rep = json.load(open(ident))
            json.dump(dict(rep, basis_sha1="000000000000"),
                      open(ident, "w"))
            try:
                kpf.build_arm("cpu", basis, "z", torch,
                              identity_report=ident)
                raise AssertionError("принят чужой отчёт гейта")
            except SystemExit as e:
                assert "другого файла" in str(e), e
    finally:
        k15_context.build, sys.argv, kpd.gate_paths = saved
    print("интеграция k15f пройдена: кэш, предобучение, гейт, рука")
    return 0


if __name__ == "__main__":
    if "--integration" in sys.argv:
        sys.exit(integration())
    sys.exit(main())
