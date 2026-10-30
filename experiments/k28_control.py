#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-28: малый контроллер со свежим состоянием — одна реализация для
обучения (GPU) и раскатки (CPU float32).

Ошибка шага j (опорная поза ТЕКУЩЕГО шага j, не j+1):
  e_p = (p_ref(j) − p_now) / σ_p,  e_R = Log(R_ref(j) R_nowᵀ) / σ_R;
  вход сети — e, обрезанная поканально в [−10, 10] (необрезанная норма
  сохраняется для отчёта).
Поправка: f_θ: [c_j (94), e_j (6)] -> 128 -> 128 -> 6, GELU; последний
  Linear без bias, нулевой при инициализации; внутренние — детерминированные
  ненулевые (U(±1/√fan_in) из локального генератора);
  r_j = f_θ(c_j, e_j) − f_θ(c_j, 0) (два вызова одной формы — при e = 0
  поправка ровно 0); Δa = κ ⊙ tanh(r), κ = 0.2 · (U − L) / 2 поканально;
  a = clip(a0 + Δa, L, U). Схват не корректируется (знак логита базы).
Всё — float32 вне autocast. Runtime — CPU-исполнитель шагов по плану:
  проверяет план (id, формы, конечность), шаг j ∈ [0, 19] строго по порядку
  для каждой среды, активность среды; при ошибке ничего не меняет.
"""
import hashlib
import math
import sys

import numpy as np

CTX, E_DIM, HID = 94, 6, 128
KAPPA_FRAC = 0.2
E_CLIP = 10.0
T = 20
VERSION = "fb1"
CONFIG = dict(version=VERSION, ctx=CTX, e=E_DIM, hidden=HID,
              kappa="0.2*(U-L)/2", e_clip=E_CLIP,
              residual="f(c,e) − f(c,0), last layer zero, no bias",
              error="e_p=(p_ref(j)−p_now)/σ_p, e_R=Log(R_ref(j) R_nowᵀ)/σ_R",
              grip="не корректируется")


def build(torch):
    import k28_geometry as kg
    nn = torch.nn
    G = kg.build(torch)

    def uniform_(t, bound, gen):
        with torch.no_grad():
            t.copy_((torch.rand(t.shape, generator=gen) * 2 - 1) * bound)

    class Controller(nn.Module):
        def __init__(self, gen):
            super().__init__()
            self.l1 = nn.Linear(CTX + E_DIM, HID)
            self.l2 = nn.Linear(HID, HID)
            self.l3 = nn.Linear(HID, E_DIM, bias=False)
            for m in (self.l1, self.l2):
                b = 1.0 / math.sqrt(m.in_features)
                uniform_(m.weight, b, gen)
                uniform_(m.bias, b, gen)
            with torch.no_grad():
                self.l3.weight.zero_()

        def f(self, x):
            g = nn.functional.gelu
            return self.l3(g(self.l2(g(self.l1(x)))))

        def residual(self, c, e):
            dev = c.device.type
            with torch.autocast(dev, enabled=False):
                c, e = c.float(), e.float()
                return self.f(torch.cat([c, e], -1)) - \
                    self.f(torch.cat([c, torch.zeros_like(e)], -1))

    def errors(ref_p, ref_R, p_now, R_now, sigma_p, sigma_R):
        """-> (e обрезанная [..., 6], необрезанная норма [...])."""
        dev = ref_p.device.type
        with torch.autocast(dev, enabled=False):
            sp = torch.as_tensor(sigma_p, dtype=torch.float32,
                                 device=ref_p.device)
            ep = (ref_p.float() - p_now.float()) / sp
            eR = G["rot_err"](ref_R.float(), R_now.float()) / float(sigma_R)
            e = torch.cat([ep, eR], -1)
            return e.clamp(-E_CLIP, E_CLIP), e.norm(dim=-1)

    def kappa(L, U, device):
        L = torch.as_tensor(np.asarray(L)[:6], dtype=torch.float32,
                            device=device)
        U = torch.as_tensor(np.asarray(U)[:6], dtype=torch.float32,
                            device=device)
        return KAPPA_FRAC * (U - L) / 2, L, U

    def correct(a0, r, L, U):
        """a0 [..., 6] (единицы кодека), r [..., 6] -> a, Δa, флаги."""
        dev = a0.device.type
        with torch.autocast(dev, enabled=False):
            k, Lt, Ut = kappa(L, U, a0.device)
            da = k * torch.tanh(r.float())
            raw = a0.float() + da
            a = torch.minimum(torch.maximum(raw, Lt), Ut)
            return a, da, dict(clip=(raw != a), kappa_sat=(
                torch.tanh(r.float()).abs() > 0.99))

    class Runtime:
        """Исполнение 8 лёгких шагов по плану на CPU. plan — словарь
        CPU-тензоров (plan_id, a0 [B, 20, 6], grip [B, 20], ref_p [B, 21, 3],
        ref_R [B, 21, 3, 3], c [B, 20, 94], sigma_p, sigma_R, L, U)."""

        def __init__(self, ctrl_state):
            self.ctrl = Controller(torch.Generator().manual_seed(0))
            self.ctrl.load_state_dict({k: v.detach().cpu().float()
                                       for k, v in ctrl_state.items()})
            self.ctrl.eval()
            for p in self.ctrl.parameters():
                p.requires_grad_(False)
            self.plan_id = None
            self.last = None

        def weights_hash(self):
            h = hashlib.sha1()
            for k, v in sorted(self.ctrl.state_dict().items()):
                h.update(k.encode())
                h.update(np.ascontiguousarray(v.numpy()).tobytes())
            return h.hexdigest()

        def begin(self, plan):
            B = plan["a0"].shape[0]
            want = dict(a0=(B, T, 6), grip=(B, T), ref_p=(B, T + 1, 3),
                        ref_R=(B, T + 1, 3, 3), c=(B, T, CTX))
            for k, s in want.items():
                if tuple(plan[k].shape) != s:
                    raise SystemExit(f"план: {k} формы "
                                     f"{tuple(plan[k].shape)}, нужно {s}")
                if not bool(torch.isfinite(plan[k]).all()):
                    raise SystemExit(f"план: {k} не конечен")
            if plan["plan_id"] == self.plan_id:
                raise SystemExit("план с тем же plan_id уже исполнялся")
            self.plan_id = plan["plan_id"]
            self.last = [-1] * B

        def step(self, plan, state8, j, active):
            """state8 [B, 8] свежие состояния, j — шаг плана, active [B]
            -> (a [B, 7] единицы кодека, dbg). Ошибки — до изменения
            состояния исполнителя."""
            B = plan["a0"].shape[0]
            if plan["plan_id"] != self.plan_id:
                raise SystemExit("устаревший план")
            if not (0 <= int(j) < T):
                raise SystemExit(f"шаг {j} вне [0, {T - 1}]")
            s = torch.as_tensor(np.asarray(state8, np.float32))
            if tuple(s.shape) != (B, 8) or not bool(torch.isfinite(s).all()):
                raise SystemExit(f"состояние формы {tuple(s.shape)} или "
                                 f"не конечно")
            act = torch.as_tensor(np.asarray(active, bool))
            if tuple(act.shape) != (B,):
                raise SystemExit("маска активности неверной формы")
            bad = [b for b in range(B) if bool(act[b]) and
                   self.last[b] != int(j) - 1]
            if bad:
                raise SystemExit(f"шаг {j} не по порядку для сред {bad} "
                                 f"(последний {[self.last[b] for b in bad]})")
            with torch.no_grad():
                p_now, R_now = G["pose_of_state8"](s)
                e, en = errors(plan["ref_p"][:, j], plan["ref_R"][:, j],
                               p_now, R_now, plan["sigma_p"],
                               plan["sigma_R"])
                r = self.ctrl.residual(plan["c"][:, j], e)
                a, da, fl = correct(plan["a0"][:, j], r, plan["L"],
                                    plan["U"])
                a = torch.where(act[:, None], a, plan["a0"][:, j].float())
                da = torch.where(act[:, None], da, torch.zeros_like(da))
            for b in range(B):
                if bool(act[b]):
                    self.last[b] = int(j)
            out = torch.cat([a, plan["grip"][:, j, None].float()], -1)
            return out.numpy(), dict(e=e.numpy(), e_norm=en.numpy(),
                                     r=r.numpy(), da=da.numpy(),
                                     a0=plan["a0"][:, j].numpy(),
                                     clip=fl["clip"].numpy(),
                                     kappa_sat=fl["kappa_sat"].numpy())

    return dict(Controller=Controller, errors=errors, correct=correct,
                kappa=kappa, Runtime=Runtime, G=G)


def fake_plan(torch, B=3, seed=0, plan_id="p0"):
    """Игрушечный план (для самопроверок модулей K-28)."""
    import k28_geometry as kg
    G = kg.build(torch)
    g = torch.Generator().manual_seed(seed)
    ref_p = torch.cumsum(torch.randn(B, T + 1, 3, generator=g) * 0.01, 1)
    ref_R = G["exp_so3"](torch.cumsum(torch.randn(B, T + 1, 3, generator=g)
                                      * 0.02, 1))
    L = -np.ones(7) * 0.9
    U = np.ones(7) * 0.9
    return dict(plan_id=plan_id,
                a0=torch.rand(B, T, 6, generator=g) * 1.2 - 0.6,
                grip=torch.where(torch.rand(B, T, generator=g) > 0.5, 1.0,
                                 -1.0), ref_p=ref_p, ref_R=ref_R,
                c=torch.randn(B, T, CTX, generator=g),
                sigma_p=np.array([0.01, 0.01, 0.01], np.float32),
                sigma_R=0.02, L=L, U=U)


def selftest():
    import torch
    C = build(torch)
    G = C["G"]
    ctrl = C["Controller"](torch.Generator().manual_seed(3))
    plan = fake_plan(torch)
    B = 3
    # нулевой последний слой -> поправка 0; нулевая ошибка -> ровно 0 при
    # любых весах
    c = plan["c"][:, 0]
    e = torch.randn(B, 6)
    assert float(ctrl.residual(c, e).abs().max()) == 0.0
    with torch.no_grad():
        ctrl.l3.weight.normal_(0, 1)
    assert float(ctrl.residual(c, torch.zeros(B, 6)).abs().max()) == 0.0
    r = ctrl.residual(c, e)
    assert float(r.abs().max()) > 0
    # пределы и κ
    a, da, fl = C["correct"](plan["a0"][:, 0], r * 100, plan["L"], plan["U"])
    k, Lt, Ut = C["kappa"](plan["L"], plan["U"], torch.device("cpu"))
    assert (a >= Lt - 1e-7).all() and (a <= Ut + 1e-7).all()
    assert (da.abs() <= k + 1e-7).all()
    # ошибка: опорная поза j против свежей; изменение свежего состояния
    # меняет вход и (на ненулевом контроллере) выход
    p_now, R_now = plan["ref_p"][:, 0].clone(), plan["ref_R"][:, 0].clone()
    e0, n0 = C["errors"](plan["ref_p"][:, 0], plan["ref_R"][:, 0], p_now,
                         R_now, plan["sigma_p"], plan["sigma_R"])
    assert float(e0.abs().max()) < 1e-5
    e1, n1 = C["errors"](plan["ref_p"][:, 0], plan["ref_R"][:, 0],
                         p_now + 0.5, R_now, plan["sigma_p"], plan["sigma_R"])
    assert float(e1.abs().max()) == E_CLIP and float(n1.min()) > E_CLIP
    # Runtime: шаги по порядку, схват не меняется, ошибки до изменения
    rt = C["Runtime"](ctrl.state_dict())
    rt.begin(plan)
    aa = torch.as_tensor(np.asarray(plan["a0"][:, 0]))
    state = np.concatenate([plan["ref_p"][:, 0].numpy(),
                            G["log_so3"](plan["ref_R"][:, 0]).numpy(),
                            np.zeros((B, 2))], 1)
    out, dbg = rt.step(plan, state, 0, np.ones(B, bool))
    assert out.shape == (B, 7) and np.array_equal(
        out[:, 6], plan["grip"][:, 0].numpy())
    assert np.allclose(out[:, :6], aa.numpy(), atol=1e-5)  # ошибка ≈ 0
    st2 = state.copy()
    st2[:, 0] += 0.003
    out2, dbg2 = rt.step(plan, st2, 1, np.array([True, True, False]))
    assert not np.allclose(out2[:2, :6], plan["a0"][:2, 1].numpy())
    assert np.array_equal(out2[2, :6], plan["a0"][2, 1].numpy())
    snap = list(rt.last)
    for bad in (lambda: rt.step(plan, st2, 1, np.ones(B, bool)),     # повтор
                lambda: rt.step(plan, st2, 5, np.ones(B, bool)),     # пропуск
                lambda: rt.step(plan, st2, 20, np.ones(B, bool)),    # вне
                lambda: rt.step(plan, np.full((B, 8), np.nan), 2,
                                np.ones(B, bool)),
                lambda: rt.step(plan, st2[:2], 2, np.ones(B, bool)),
                lambda: rt.step(dict(plan, plan_id="старый"), st2, 2,
                                np.ones(B, bool))):
        try:
            bad()
            raise AssertionError("ошибка исполнения принята")
        except SystemExit:
            pass
        assert rt.last == snap
    try:
        rt.begin(plan)
        raise AssertionError("повторный plan_id принят")
    except SystemExit:
        pass
    # CPU (Runtime) = функции обучения на тех же входах
    with torch.no_grad():
        p_n, R_n = G["pose_of_state8"](torch.as_tensor(st2, dtype=torch.float32))
        e_t, _n = C["errors"](plan["ref_p"][:, 1], plan["ref_R"][:, 1], p_n,
                              R_n, plan["sigma_p"], plan["sigma_R"])
        a_t, _d, _f = C["correct"](plan["a0"][:, 1], ctrl.residual(
            plan["c"][:, 1], e_t), plan["L"], plan["U"])
    assert np.allclose(a_t[:2].numpy(), out2[:2, :6], atol=1e-6)
    # bf16-autocast не меняет контроллер
    with torch.autocast("cpu", dtype=torch.bfloat16):
        rb = ctrl.residual(c, e)
    assert rb.dtype == torch.float32 and torch.equal(rb, ctrl.residual(c, e))
    print("самопроверка k28_control пройдена: нулевой последний слой и "
          "нулевая ошибка -> поправка ровно 0, пределы L/U и κ, ошибка по "
          "опорной позе j и свежему состоянию (обрезка ±10), схват не "
          "меняется, неактивная среда — номинал, шаги строго по порядку, "
          "отказы (повтор, пропуск, вне [0, 19], NaN, форма, устаревший "
          "план, повторный plan_id) без изменения исполнителя, CPU-путь = "
          "функциям обучения, float32 вне autocast")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль; самопроверка: --selftest")
