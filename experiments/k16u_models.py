#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-16u: пара «предсказатель цели G + корректор C» с маршрутизацией
градиентов (режим обучения, не новая архитектура).

G — GoalNet K-16 (инициализация — goal_hist из data/k16/models.pt);
C — CtxHead K-16b (ширина и слои без изменений): ctx (богатый контекст +
скрытый вектор G) или ctx_goal (то же + явная предсказанная цель G).

РЕЖИМЫ:
  frozen_goal    G заморожен (requires_grad=False), выходы отсоединены;
  separate_goal  G обучается только по своей ошибке цели; скрытый вектор
                 и явная цель отсоединяются перед подачей в C;
  joint_goal     выходы G не отсоединяются: ошибки C проходят в G;
  joint_ctx      то же для головы ctx.
Корректор всегда получает СОБСТВЕННОЕ предсказание G, вычисленное
текущими весами в этом же батче.
"""
import copy

MODES = {"frozen_goal": (True, "frozen"), "separate_goal": (True, "separate"),
         "joint_goal": (True, "joint"), "joint_ctx": (False, "joint")}


def build(torch, d_model=768, n_pos=16):
    import k16b_train as bt
    nn = torch.nn
    C, CtxHead = bt.build_heads(torch, d_model, n_pos)

    class Pair(nn.Module):
        def __init__(self, mode, g_state, hidden=bt.HIDDEN, c_seed=0):
            super().__init__()
            if mode not in MODES:
                raise ValueError(f"режим {mode!r}")
            use_goal, route = MODES[mode]
            self.mode, self.route = mode, route
            self.G = C["GoalNet"]()
            self.G.load_state_dict(copy.deepcopy(g_state))
            torch.manual_seed(c_seed)            # одинаковая инициализация C
            self.C = CtxHead(hidden, use_goal=use_goal)
            if route == "frozen":
                for p in self.G.parameters():
                    p.requires_grad_(False)

        def forward(self, h18, a0, sf):
            hid = self.G.hidden(h18, a0, sf)
            g = self.G.body[-1](hid)
            if self.route in ("frozen", "separate"):
                hid_in, g_in = hid.detach(), g.detach()
            else:
                hid_in, g_in = hid, g
            r, aux = self.C(h18, a0, sf, torch.cat([hid_in, g_in], 1))
            return dict(r=r, aux=aux, goal=g, hid=hid)
    return Pair
