#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15e: семейство кандидатов a0 + α·δ поверх замороженных q0 и h18.

ЕДИНСТВЕННАЯ функция построения кандидатов для всего K-15e: её вызывают
рука роллаута, будущий сборщик веток, проверка вывода и тренер. Две разные
реализации однажды разошлись бы, и критик учился бы не на тех действиях,
что исполняются.

ОПРЕДЕЛЕНИЕ (план K-15e §2.3). δ — поправка h18 к a0 на исполняемых шагах:
  рука   a_α = a0 + α·(a1 − a0);
  схват  ℓ_α = ℓ0 + α·(ℓ1 − ℓ0),  g_α = g0 + tanh(ℓ_α/2) − tanh(ℓ0/2)
         — масштабируется приращение ЛОГИТА, а не итоговая команда, ровно
         в той параметризации, в которой h18 его выдаёт (k15d_depth_refine,
         DepthRefiner.combine);
  хвост  чанка за H_EXEC равен a0 (у h18 он тоже равен a0).

ТОЧКИ α=0 И α=1 ПОДСТАВЛЯЮТСЯ, А НЕ ВЫЧИСЛЯЮТСЯ: a0 + 1.0·(a1 − a0)
побитово не равно a1. Поэтому α=0 побитово воспроизводит a0, α=1 — выход
h18. Это проверяется самопроверкой и каждым вызовом руки.

КОНТРОЛЬ НАПРАВЛЕНИЯ. Отрицательные α — поправка той же величины в
обратную сторону. Оракул по ним измеряет выигрыш от одного лишь
разнообразия траекторий: в K-15d h1 отличался от h18 на ~1e-3 на шаг, а
исходы расходились в 40 % кластеров. Полезность направления δ — это
разница оракулов «по направлению» и «против направления», а не
абсолютный оракул.
"""
import argparse
import math
import sys

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
CONTROL_ALPHAS = (-0.5, -1.0)
GRIP_EPS = 1e-3            # как k15d_depth_refine.GRIP_EPS
ACTION_CLIP_BOUND = 1.5


def alpha_label(alpha):
    """Метка руки: 0.25 -> a025, -0.5 -> am050."""
    v = int(round(abs(float(alpha)) * 100))
    return ("am" if float(alpha) < 0 else "a") + f"{v:03d}"


def label_alpha(label):
    s = str(label)
    if s.startswith("am"):
        return -int(s[2:]) / 100.0
    if s.startswith("a") and s[1:].isdigit():
        return int(s[1:]) / 100.0
    raise ValueError(f"метка {label!r} не описывает α")


def make_candidates(a0, a1, lg0, lg1, alphas, h_exec):
    """Кандидаты [B, K, T, 7] для списка α.

    a0, a1 — [B, T, 7] (q0 и h18), lg0, lg1 — [B, H_EXEC] (логиты схвата
    q0 и h18). Ничего не обучается: функция чистая и дифференцируема по
    входам, если понадобится.
    """
    import torch
    if a0.shape != a1.shape or a0.ndim != 3 or a0.shape[-1] != 7:
        raise ValueError(f"формы a0 {tuple(a0.shape)} и a1 {tuple(a1.shape)}")
    h = int(h_exec)
    if lg0.shape != (a0.shape[0], h) or lg1.shape != lg0.shape:
        raise ValueError("логиты схвата должны быть [B, H_EXEC]")
    if not torch.equal(a1[:, h:], a0[:, h:]):
        raise ValueError("хвост h18 за H_EXEC обязан совпадать с a0")
    out = []
    for al in alphas:
        al = float(al)
        if al == 0.0:
            out.append(a0.clone())
            continue
        if al == 1.0:
            out.append(a1.clone())
            continue
        arm = a0[:, :h, :6] + al * (a1[:, :h, :6] - a0[:, :h, :6])
        lg = lg0 + al * (lg1 - lg0)
        g = a0[:, :h, 6] + (torch.tanh(lg * 0.5) - torch.tanh(lg0 * 0.5))
        exec_part = torch.cat([arm, g[..., None]], dim=-1)
        out.append(torch.cat([exec_part, a0[:, h:]], dim=1))
    return torch.stack(out, dim=1)


def selftest():
    import torch
    g = torch.Generator().manual_seed(0)
    B, T, H = 4, 20, 8
    a0 = torch.rand(B, T, 7, generator=g) * 1.6 - 0.8
    lg0 = 2.0 * torch.atanh(a0[:, :H, 6].clamp(-1 + GRIP_EPS, 1 - GRIP_EPS))
    r = torch.randn(B, H, 7, generator=g)
    lg1 = lg0 + 3.0 * torch.tanh(r[..., 6])
    a1 = a0.clone()
    a1[:, :H, :6] = a0[:, :H, :6] + 0.3 * torch.tanh(r[..., :6])
    a1[:, :H, 6] = a0[:, :H, 6] + (torch.tanh(lg1 * 0.5)
                                   - torch.tanh(lg0 * 0.5))
    al = ALPHAS + CONTROL_ALPHAS
    c = make_candidates(a0, a1, lg0, lg1, al, H)
    assert c.shape == (B, len(al), T, 7)
    assert torch.equal(c[:, 0], a0), "α=0 обязан побитово дать a0"
    assert torch.equal(c[:, al.index(1.0)], a1), "α=1 обязан дать h18"
    # хвост неподвижен у всех
    for k in range(len(al)):
        assert torch.equal(c[:, k, H:], a0[:, H:])
    # рука линейна по α, контроль — зеркало
    d = a1[:, :H, :6] - a0[:, :H, :6]
    half = c[:, al.index(0.5), :H, :6]
    assert torch.allclose(half, a0[:, :H, :6] + 0.5 * d, atol=1e-6)
    neg = c[:, al.index(-1.0), :H, :6]
    assert torch.allclose(neg, a0[:, :H, :6] - d, atol=1e-6)
    # схват ограничен при любом α: |g - (g0 - tanh(ℓ0/2))| < 1
    off = a0[:, :H, 6] - torch.tanh(lg0 * 0.5)
    for k in range(len(al)):
        assert bool(((c[:, k, :H, 6] - off).abs() < 1.0 + 1e-6).all())
    # КОНТРОЛЬ ВЫРОЖДЕННОСТИ: вычисленное a0 + 1·(a1-a0) побитово НЕ обязано
    # совпасть с a1 — поэтому α=1 и подставляется
    calc = a0[:, :H, :6] + 1.0 * (a1[:, :H, :6] - a0[:, :H, :6])
    assert torch.allclose(calc, a1[:, :H, :6], atol=1e-6)
    # ошибки входа
    for bad in (lambda: make_candidates(a0, a1[:, :10], lg0, lg1, al, H),
                lambda: make_candidates(a0, a1, lg0[:, :3], lg1, al, H)):
        try:
            bad()
            raise AssertionError("принят неверный вход")
        except ValueError:
            pass
    a1_bad = a1.clone()
    a1_bad[:, H:] += 1.0
    try:
        make_candidates(a0, a1_bad, lg0, lg1, al, H)
        raise AssertionError("принят h18 с подвижным хвостом")
    except ValueError:
        pass
    assert [alpha_label(x) for x in al] == ["a000", "a025", "a050", "a075",
                                            "a100", "am050", "am100"]
    assert all(math.isclose(label_alpha(alpha_label(x)), x) for x in al)
    print("самопроверка k15e_candidates пройдена")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="K-15e: кандидаты a0 + α·δ")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    raise SystemExit("это модуль")
