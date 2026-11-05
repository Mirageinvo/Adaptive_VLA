#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-32: сырой вектор состояния, признаки actor/critic, замороженная
нормировка, actor (общий узловой MLP) и critic.

Сырой вектор состояния S (6H + 72), из него однозначно собираются все входы:
  [mean x0, mean x1, mean x2 (3H), x0_root*, x1_mid*, x2_leaf* (3H),
   s11n (11), elapsed/600 (1), Z0[:8]/half (48), центр/20 и ширина/20 трёх
   узлов (6), масса трёх узлов на префиксе (3), c трёх узлов (3)].
Actor x_r (4H + 67): [3 средних (3H), hidden узла r (H), s11n, elapsed/600,
  Z0[:8]/half, one-hot уровня (3), центр/ширина узла r /20 (2), масса r,
  c_r]. Critic — весь S (6H + 72).
Нормировка (своя на сид, замороженная): mean/std float64, std < 1e−3 ->
  1, clip ±5; one-hot и elapsed/600 — без стандартизации. Статистики actor —
  по трём узлам каждой строки с равным весом; critic — по строкам.
Actor: (4H+67) -> 64 -> 32 -> (1 + d), GELU; один MLP на три узла; выход 0 —
  logit маршрута, 1… — raw μ. Critic: (6H+72) -> 128 -> 64 -> 1, GELU,
  sigmoid. Последний Linear — нули (p = 1/3, μ = 0, V = 0.5). Инициализация
  локальная: генератор от (seed, имя тензора); глобальный RNG не трогается.
"""
import hashlib
import math
import sys

import numpy as np

STD_FLOOR = 1e-3
CLIP = 5.0
ACTOR_HID = (64, 32)
CRITIC_HID = (128, 64)


def layout(H):
    o = {}
    i = 0
    for k, n in (("means", 3 * H), ("nodes", 3 * H), ("s11n", 11),
                 ("elapsed", 1), ("u0", 48), ("cw", 6), ("mass", 3),
                 ("c", 3)):
        o[k] = (i, i + n)
        i += n
    o["size"] = i
    return o


def state_vector(F1, s11n, elapsed, Z0, half, pinfo):
    """Одно наблюдение -> S [6H+72] float32 numpy."""
    x0 = F1["x0"][0].float().cpu().numpy()
    x1 = F1["x1"][0].float().cpu().numpy()
    x2 = F1["x2"][0].float().cpu().numpy()
    pth = pinfo["path"]
    u0 = (np.asarray(Z0, np.float32)[:8] / np.asarray(half, np.float32))
    S = np.concatenate([x0.mean(0), x1.mean(0), x2.mean(0),
                        x0[pth[0]], x1[pth[1]], x2[pth[2]],
                        np.asarray(s11n, np.float32).ravel(),
                        [float(elapsed) / 600.0], u0.ravel(),
                        (np.asarray(pinfo["cw"], np.float64) / 20.0).ravel(),
                        np.asarray(pinfo["mass"], np.float64),
                        np.asarray(pinfo["c"], np.float64)])
    return S.astype(np.float32)


def actor_inputs_np(S, H):
    """S [N, 6H+72] -> X [N, 3, 4H+67] (сырые)."""
    L = layout(H)
    S = np.asarray(S, np.float64)
    N = len(S)
    out = np.zeros((N, 3, 4 * H + 67))
    nodes = S[:, L["nodes"][0]:L["nodes"][1]].reshape(N, 3, H)
    for r in range(3):
        hot = np.zeros((N, 3))
        hot[:, r] = 1.0
        cw = S[:, L["cw"][0]:L["cw"][1]].reshape(N, 3, 2)[:, r]
        out[:, r] = np.concatenate([
            S[:, L["means"][0]:L["means"][1]], nodes[:, r],
            S[:, L["s11n"][0]:L["s11n"][1]],
            S[:, L["elapsed"][0]:L["elapsed"][1]],
            S[:, L["u0"][0]:L["u0"][1]], hot, cw,
            S[:, L["mass"][0] + r:L["mass"][0] + r + 1],
            S[:, L["c"][0] + r:L["c"][0] + r + 1]], 1)
    return out


def actor_raw_mask(H):
    """Поля actor без стандартизации: elapsed и one-hot."""
    m = np.zeros(4 * H + 67, bool)
    m[4 * H + 11] = True                                  # elapsed
    m[4 * H + 12 + 48:4 * H + 12 + 48 + 3] = True         # one-hot
    return m


def critic_raw_mask(H):
    L = layout(H)
    m = np.zeros(L["size"], bool)
    m[L["elapsed"][0]] = True
    return m


def fit_normalizer(S_rows, H):
    """S_rows [n, 6H+72] (демонстрационные строки, elapsed = 0)."""
    S = np.asarray(S_rows, np.float64)
    if not np.isfinite(S).all():
        raise SystemExit("нормировка: нечисловые признаки")
    X = actor_inputs_np(S, H).reshape(-1, 4 * H + 67)     # 3 узла на строку
    out = {}
    for nm, A, mask in (("actor", X, actor_raw_mask(H)),
                        ("critic", S, critic_raw_mask(H))):
        mu = A.mean(0)
        sd = A.std(0)
        sd = np.where(sd < STD_FLOOR, 1.0, sd)
        mu[mask], sd[mask] = 0.0, 1.0
        out[f"{nm}_mean"], out[f"{nm}_std"] = mu, sd
    out["H"] = int(H)
    out["n_rows"] = int(len(S))
    return out


def norm_sha(N):
    h = hashlib.sha1()
    for k in ("actor_mean", "actor_std", "critic_mean", "critic_std"):
        h.update(np.ascontiguousarray(N[k], np.float64).tobytes())
    return h.hexdigest()


def build(torch, H, d, seed, norm):
    nn = torch.nn

    def local_init(mod, tag):
        for name, p in mod.named_parameters():
            key = hashlib.sha1(f"k32|{seed}|{tag}|{name}".encode()).digest()
            g = torch.Generator().manual_seed(int.from_bytes(key[:8], "little")
                                              & ((1 << 63) - 1))
            with torch.no_grad():
                if name.startswith("out."):
                    p.zero_()
                else:
                    fan_in = p.shape[-1] if p.dim() > 1 else \
                        dict(mod.named_parameters())[
                            name.replace("bias", "weight")].shape[-1]
                    b = 1.0 / math.sqrt(fan_in)
                    p.copy_((torch.rand(p.shape, generator=g) * 2 - 1) * b)

    an = {k: torch.as_tensor(np.asarray(norm[k], np.float32))
          for k in ("actor_mean", "actor_std", "critic_mean", "critic_std")}

    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(4 * H + 67, ACTOR_HID[0])
            self.l2 = nn.Linear(*ACTOR_HID)
            self.out = nn.Linear(ACTOR_HID[1], 1 + d)
            self.register_buffer("mean", an["actor_mean"].clone())
            self.register_buffer("std", an["actor_std"].clone())
            self.d = d

        def forward(self, X):
            """X [N, 3, 4H+67] сырые -> logits [N,3], raw_mu [N,3,d]."""
            z = torch.clamp((X - self.mean) / self.std, -CLIP, CLIP)
            g_ = nn.functional.gelu
            y = self.out(g_(self.l2(g_(self.l1(z)))))
            return y[..., 0], y[..., 1:]

    class Critic(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(6 * H + 72, CRITIC_HID[0])
            self.l2 = nn.Linear(*CRITIC_HID)
            self.out = nn.Linear(CRITIC_HID[1], 1)
            self.register_buffer("mean", an["critic_mean"].clone())
            self.register_buffer("std", an["critic_std"].clone())

        def forward(self, S):
            z = torch.clamp((S - self.mean) / self.std, -CLIP, CLIP)
            g_ = nn.functional.gelu
            return torch.sigmoid(self.out(g_(self.l2(g_(self.l1(z))))))[:, 0]
    with torch.random.fork_rng(devices=[]):
        a, c = Actor(), Critic()
    local_init(a, "actor")
    local_init(c, "critic")
    return a, c


def actor_inputs(torch, S, H):
    """Тот же сбор, что actor_inputs_np, в torch (для PPO)."""
    L = layout(H)
    N = S.shape[0]
    nodes = S[:, L["nodes"][0]:L["nodes"][1]].reshape(N, 3, H)
    cw = S[:, L["cw"][0]:L["cw"][1]].reshape(N, 3, 2)
    eye = torch.eye(3, dtype=S.dtype, device=S.device)
    parts = [S[:, None, L["means"][0]:L["means"][1]].expand(N, 3, 3 * H),
             nodes,
             S[:, None, L["s11n"][0]:L["elapsed"][1]].expand(N, 3, 12),
             S[:, None, L["u0"][0]:L["u0"][1]].expand(N, 3, 48),
             eye[None].expand(N, 3, 3), cw,
             S[:, L["mass"][0]:L["mass"][1]][..., None],
             S[:, L["c"][0]:L["c"][1]][..., None]]
    return torch.cat(parts, -1)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def selftest():
    import torch
    rng = np.random.default_rng(0)
    H = 16
    L = layout(H)
    assert L["size"] == 6 * H + 72
    S = rng.normal(size=(40, L["size"])).astype(np.float32)
    S[:, L["elapsed"][0]] = 0.0
    N = fit_normalizer(S, H)
    Xn = actor_inputs_np(S, H)
    Xt = actor_inputs(torch, torch.as_tensor(S, dtype=torch.float64), H)
    assert np.array_equal(Xn, Xt.numpy()), "сбор признаков np ≠ torch"
    assert Xn.shape == (40, 3, 4 * H + 67)
    # one-hot и elapsed без стандартизации
    m = actor_raw_mask(H)
    assert (Xn[:, :, m][:, :, 1:].sum(-1) == 1).all()
    assert N["actor_mean"][m].tolist() == [0.0] * 4 and \
        (N["actor_std"][m] == 1).all()
    ts = torch.random.get_rng_state().clone()
    aT, cT = build(torch, H, 6, 0, N)
    aP, cP = build(torch, H, 6, 0, N)
    aF, cF = build(torch, H, 48, 0, N)
    assert torch.equal(torch.random.get_rng_state(), ts)
    sdT, sdP, sdF = aT.state_dict(), aP.state_dict(), aF.state_dict()
    assert all(torch.equal(sdT[k], sdP[k]) for k in sdT)       # T = P
    assert all(torch.equal(sdT[k], sdF[k]) for k in sdT
               if not k.startswith("out."))                    # общие слои
    assert all(torch.equal(a, b) for a, b in zip(cT.state_dict().values(),
                                                 cF.state_dict().values()))
    a1, _c = build(torch, H, 6, 1, N)
    assert not torch.equal(a1.l1.weight, aT.l1.weight)
    with torch.no_grad():
        lg, raw = aT(torch.as_tensor(Xn, dtype=torch.float32))
        v = cT(torch.as_tensor(S))
    assert torch.equal(lg, torch.zeros_like(lg)) and \
        torch.equal(raw, torch.zeros_like(raw))
    assert torch.allclose(v, torch.full_like(v, 0.5))
    # число параметров при H = 768
    H7 = 768
    N7 = dict(actor_mean=np.zeros(4 * H7 + 67), actor_std=np.ones(4 * H7 + 67),
              critic_mean=np.zeros(6 * H7 + 72),
              critic_std=np.ones(6 * H7 + 72))
    a7, c7 = build(torch, H7, 6, 0, N7)
    f7, _ = build(torch, H7, 48, 0, N7)
    assert n_params(a7) == 203271 and n_params(f7) == 204657 and \
        n_params(c7) == 607489, (n_params(a7), n_params(f7), n_params(c7))
    # clip ±5 и нечисловой вход нормировки
    big = torch.full((1, 3, 4 * H + 67), 1e6)
    with torch.no_grad():
        lg2, _ = aT(big)
    assert torch.isfinite(lg2).all()
    try:
        fit_normalizer(np.full((3, L["size"]), np.nan), H)
        raise AssertionError("нечисловая нормировка принята")
    except SystemExit:
        pass
    print("самопроверка k32_model пройдена: раскладка 6H+72 / 4H+67, сбор "
          "признаков numpy = torch, one-hot и elapsed без стандартизации, "
          "локальная инициализация (глобальный RNG не сдвинут, T = P, F/T "
          "общие слои и critic, другой seed — другие веса), нулевой выход "
          "(p = 1/3, μ = 0, V = 0.5), 203 271 / 204 657 / 607 489 параметров "
          "при H = 768, clip ±5")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    raise SystemExit("модуль; самопроверка: --selftest")
