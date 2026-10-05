#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-15d: непрерывное уточнение действия по глубине одной VLA (DAAR).

Два варианта поверх канонического q0 на слое 12:

  D0 (контроль):  a_d = a0 + Δ_d(h24, a0);  φ_d(a0) перед слоем 13,
                  LoRA слоёв 13–24.
  H1 (метод):     a1 = a0 + Δ1(h18, a0);    φ1(a0) перед слоем 13,
                  LoRA слоёв 13–18;
                  a2 = a1 + Δ2(h24, a1);    φ2(a1) перед слоем 19,
                  LoRA слоёв 19–24.

Нумерация слоёв в тексте с единицы, в коде — индексы с нуля: «слои 13–18»
это `action_expert.layers[12:18]`.

АРХИТЕКТУРНЫЕ ФАЙЛЫ НЕ ТРОГАЮТСЯ. LoRA добавляется forward-хуками на
Linear поздних слоёв action expert, а не заменой модулей: имена и
содержимое весов модели остаются прежними, и отпечаток замороженного
сравним с K-15a. Хуки действуют только пока свой уровень активен внутри
`DepthRefiner.run`, поэтому та же модель без уточнения (рука q0, гейт)
считает ровно то же, что раньше.

VLM-ПОТОК НЕ ЗАВИСИТ ОТ ACTION-ПОТОКА: маска блочного AR запрещает VLM
смотреть на action-ключи (bar.py, `_build_joint_attention_mask_blockwise_ar`).
Поэтому LoRA только в action expert, а градиент через VLM не идёт.

НУЛЕВАЯ ТОЧКА ТОЧНАЯ, А НЕ ПРИБЛИЖЁННАЯ. Последний слой каждой головы,
выход каждой φ и матрица B каждой LoRA — нули. Тогда:
  * добавка в скрытое состояние x + 0 == x побитово;
  * рука a_k = a_{k-1} + s·tanh(0) == a_{k-1} побитово;
  * схват g_k = g_{k-1} + tanh(ℓ_k/2) − tanh(ℓ_{k-1}/2), а при нулевом
    шаге ℓ_k == ℓ_{k-1}, и разность ровно ноль.

СХВАТ — ЧЕРЕЗ ЛОГИТ. Демонстрации на схвате почти ±1, и BCE на простом
аддитивном выходе тянула бы |g| за 1 без предела (на смоуке это было бы
неизбежным провалом гейта диапазона). Здесь ℓ0 = 2·atanh(clip(g0)),
ℓ_k = ℓ_{k-1} + s_ℓ·tanh(r), исполняемое g_k телескопически равно
g0 + tanh(ℓ_k/2) − tanh(ℓ0/2), то есть ограничено интервалом ширины 2 вокруг
g0 − tanh(ℓ0/2) ≈ 0. Обрезка применяется только к постоянному входу g0,
не к обучаемому пути.

ГОЛОВЫ ВЫДАЮТ ПОПРАВКУ ТОЛЬКО ДЛЯ ИСПОЛНЯЕМЫХ ШАГОВ (H_EXEC = 8). Остальные
12 позиций чанка равны a0: на них нет ни цели отбора, ни исполнения, и
оставить их неподвижными честнее, чем отдавать их необученному выходу.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import math
import os
import sys
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

VARIANTS = ("d0", "h1")
# Уровни: имя -> (первый слой, слой после последнего), индексы с нуля.
LEVELS = {
    "d0": (("d", 12, 24),),
    "h1": (("1", 12, 18), ("2", 18, 24)),
}
# Фаза -> уровень, который она обучает. Фаза 2 H1 требует замороженную 1.
PHASES = {"d0": "d", "h1p1": "1", "h1p2": "2"}
PHASE_VARIANT = {"d0": "d0", "h1p1": "h1", "h1p2": "h1"}
LORA_TARGETS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
                "mlp.down_proj")
Q0_DEPTH = 12
GRIP_EPS = 1e-3
# Зарегистрированные гиперпараметры. Сетки до первого результата нет
# (план §2, §7): один вариант, один набор.
HP = dict(lora_rank=16, lora_alpha=16.0, fb_hidden=512, head_proj=64,
          head_hidden=1024, delta_quantile=99.0, delta_factor=2.0,
          smooth_l1_beta=1.0, grip_weight=1.0)
FB_MODES = ("normal", "zero", "shuffle")


def grip_logit(g: torch.Tensor) -> torch.Tensor:
    """ℓ = 2·atanh(g) с обрезкой ВХОДА: g — постоянный вход, не выход."""
    c = 1.0 - GRIP_EPS
    return 2.0 * torch.atanh(g.float().clamp(-c, c))


class LoRA(nn.Module):
    """y += (x A^T) B^T · alpha/r. B нулевая: в нуле поправка ровно 0."""

    def __init__(self, d_in: int, d_out: int, rank: int, alpha: float):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = float(alpha) / float(rank)

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.A.dtype)
        return F.linear(F.linear(x, self.A), self.B) * self.scale


class ActionFeedback(nn.Module):
    """φ(a): чанк [B,T,7] -> добавка к action-токенам [B,P,D].

    Общий код чанка плюс обучаемая позиционная добавка: разные кодовые
    позиции получают разные проекции одного плана. Выходной слой нулевой.
    """

    def __init__(self, horizon: int, d_model: int, n_pos: int, hidden: int):
        super().__init__()
        self.inp = nn.Linear(horizon * 7, hidden)
        self.pos = nn.Parameter(torch.randn(n_pos, hidden) * 0.02)
        self.out = nn.Linear(hidden, d_model)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        e = self.inp(feat.flatten(1))[:, None, :] + self.pos[None]
        return self.out(F.gelu(e))


class DeltaHead(nn.Module):
    """Δ-логиты r(h, a_prev): [B,P,D] x [B,T,7] -> [B,H_EXEC,7].

    Своя копия финальной нормы action expert: норма Joint12 обучена читать
    h12, читать ею h18/h24 значило бы повторить ошибку провенанса HiCoRA.
    """

    def __init__(self, norm_src: nn.Module, d_model: int, n_pos: int,
                 horizon: int, h_exec: int, proj: int, hidden: int):
        super().__init__()
        self.norm = copy.deepcopy(norm_src).float()
        self.proj = nn.Linear(d_model, proj)
        self.mlp_in = nn.Linear(n_pos * proj + horizon * 7, hidden)
        self.out = nn.Linear(hidden, h_exec * 7)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.h_exec = int(h_exec)

    def forward(self, h: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        z = self.proj(self.norm(h.float())).flatten(1)
        z = F.gelu(self.mlp_in(torch.cat([z, feat.flatten(1)], dim=1)))
        return self.out(z).view(h.shape[0], self.h_exec, 7)


class LevelLoRA(nn.Module):
    """LoRA всех целевых Linear слоёв [lo, hi) одного уровня."""

    def __init__(self, layers, lo: int, hi: int, rank: int, alpha: float):
        super().__init__()
        self.lo, self.hi = int(lo), int(hi)
        self.mods = nn.ModuleDict()
        for i in range(self.lo, self.hi):
            for t in LORA_TARGETS:
                lin = resolve(layers[i], t)
                self.mods[lora_key(i, t)] = LoRA(
                    lin.in_features, lin.out_features, rank, alpha)


def lora_key(i: int, target: str) -> str:
    return f"L{i}_" + target.replace(".", "_")


def resolve(module: nn.Module, path: str) -> nn.Linear:
    m = module
    for part in path.split("."):
        m = getattr(m, part)
    if not isinstance(m, nn.Linear):
        raise TypeError(f"{path}: ожидался Linear, дано {type(m).__name__}")
    return m


class DepthRefiner(nn.Module):
    """Уточнение D0 или H1. Веса модели не принадлежат этому модулю."""

    def __init__(self, variant: str, *, norm_src: nn.Module, layers,
                 d_model: int, n_pos: int, horizon: int, h_exec: int,
                 stats: dict, hp: dict | None = None):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"вариант {variant}")
        hp = dict(HP if hp is None else hp)
        self.variant, self.hp = variant, hp
        self.d_model, self.n_pos = int(d_model), int(n_pos)
        self.horizon, self.h_exec = int(horizon), int(h_exec)
        if self.h_exec > self.horizon:
            raise ValueError("исполняемых шагов больше длины чанка")
        if len(layers) != LEVELS[variant][-1][2]:
            raise ValueError(f"слоёв {len(layers)}, уровни рассчитаны на "
                             f"{LEVELS[variant][-1][2]}")
        for key in ("delta_scale", "feat_scale", "loss_scale"):
            if key not in stats:
                raise ValueError(f"нет статистики {key}")
        ds = torch.as_tensor(stats["delta_scale"], dtype=torch.float32)
        fs = torch.as_tensor(stats["feat_scale"], dtype=torch.float32)
        ls = torch.as_tensor(stats["loss_scale"], dtype=torch.float32)
        if ds.shape != (7,) or fs.shape != (7,) or ls.shape != (6,):
            raise ValueError("формы статистики: delta 7, feat 7, loss 6")
        if not (bool((ds > 0).all()) and bool((fs > 0).all())
                and bool((ls > 0).all())):
            raise ValueError("масштабы должны быть положительными")
        self.register_buffer("delta_scale", ds)
        self.register_buffer("feat_scale", fs)
        self.register_buffer("loss_scale", ls)
        self.levels = LEVELS[variant]
        self.feedback = nn.ModuleDict()
        self.heads = nn.ModuleDict()
        self.lora = nn.ModuleDict()
        for name, lo, hi in self.levels:
            self.feedback[name] = ActionFeedback(
                self.horizon, self.d_model, self.n_pos, hp["fb_hidden"])
            self.heads[name] = DeltaHead(
                norm_src, self.d_model, self.n_pos, self.horizon,
                self.h_exec, hp["head_proj"], hp["head_hidden"])
            self.lora[name] = LevelLoRA(layers, lo, hi, hp["lora_rank"],
                                        hp["lora_alpha"])
        self._active = None
        self._handles = []
        self._attached_to = None
        # Срабатывания хуков по (уровень, слой, цель). Гейт требует ровно
        # одно на проход: так видно, что проекция вызывается как модуль, а
        # не через F.linear с весами (тогда хук молча не срабатывал бы).
        self.fire_counts = {}

    # --- хуки LoRA --------------------------------------------------------
    def attach(self, model) -> int:
        if self._handles:
            raise RuntimeError("хуки уже установлены")
        layers = model.action_expert.layers
        n = 0
        for name, lo, hi in self.levels:
            for i in range(lo, hi):
                for t in LORA_TARGETS:
                    lin = resolve(layers[i], t)
                    mod = self.lora[name].mods[lora_key(i, t)]
                    if (lin.in_features, lin.out_features) != (
                            mod.A.shape[1], mod.B.shape[0]):
                        raise RuntimeError(f"форма LoRA слоя {i} {t}")
                    self._handles.append(lin.register_forward_hook(
                        self._make_hook(name, mod, (name, i, t))))
                    n += 1
        self._attached_to = id(model)
        return n

    def detach_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        self._attached_to = None

    def _make_hook(self, level, mod, key):
        def hook(_m, inp, out):
            if self._active != level:
                return None
            self.fire_counts[key] = self.fire_counts.get(key, 0) + 1
            return out + mod.delta(inp[0]).to(out.dtype)
        return hook

    def expected_hooks(self):
        return {(n, i, t) for n, lo, hi in self.levels
                for i in range(lo, hi) for t in LORA_TARGETS}

    @contextmanager
    def active(self, level):
        prev, self._active = self._active, level
        try:
            yield
        finally:
            self._active = prev

    # --- белые списки -------------------------------------------------------
    def level_names(self):
        return [n for n, _lo, _hi in self.levels]

    def phase_named_parameters(self, phase: str):
        if PHASE_VARIANT.get(phase) != self.variant:
            raise ValueError(f"фаза {phase} не относится к {self.variant}")
        lv = PHASES[phase]
        pref = (f"feedback.{lv}.", f"heads.{lv}.", f"lora.{lv}.")
        return [(n, p) for n, p in self.named_parameters()
                if n.startswith(pref)]

    def set_phase(self, phase: str | None):
        """Обучаемым остаётся ровно белый список фазы; None — всё заморожено."""
        keep = set() if phase is None else {
            n for n, _p in self.phase_named_parameters(phase)}
        for n, p in self.named_parameters():
            p.requires_grad_(n in keep)
        return sorted(keep)

    def feat(self, a: torch.Tensor) -> torch.Tensor:
        return a.float() / self.feat_scale

    # --- прямой проход ------------------------------------------------------
    def run(self, model, *, vlm_inputs_embeds, attention_mask, position_ids,
            decode, train_level=None, stop_after=None, fb_mode=None,
            replace_prev=None, keep_hidden=False):
        """Один проход q0 -> уровни. Возвращает q0, a0 и действия уровней.

        train_level: уровень, по которому строится граф; всё до него идёт
          без градиента (там нет обучаемого), всё после не считается.
        fb_mode: {уровень: "normal"|"zero"|"shuffle"} — диагностика
          причинности; shuffle подаёт в φ план соседней строки батча, база
          сложения и признаки головы остаются своими.
        replace_prev: {уровень: чанк [B,T,7]} — предыдущий план уровня
          заменяется данным чанком целиком (база, φ, признаки головы);
          диагностика подставляет туда демонстрацию.
        """
        if self._attached_to != id(model):
            raise RuntimeError("хуки LoRA не установлены на эту модель")
        names = self.level_names()
        if train_level is not None and train_level not in names:
            raise ValueError(f"уровень {train_level}")
        if stop_after is None:
            stop_after = train_level if train_level is not None else names[-1]
        fb_mode = dict(fb_mode or {})
        replace_prev = dict(replace_prev or {})
        for k, v in fb_mode.items():
            if k not in names or not (isinstance(v, torch.Tensor)
                                      or v in FB_MODES):
                raise ValueError(f"fb_mode {k}={v!r}")
        for k in replace_prev:
            if k not in names:
                raise ValueError(f"replace_prev {k}")

        batch = int(vlm_inputs_embeds.shape[0])
        device, dtype = vlm_inputs_embeds.device, vlm_inputs_embeds.dtype
        n_pos = int(model.block_size)
        if n_pos != self.n_pos:
            raise RuntimeError(f"позиций {n_pos}, уточнение на {self.n_pos}")
        layers = model.action_expert.layers

        def run_layers(vlm, act, lo, hi):
            for i in range(lo, hi):
                vlm, act = model._shared_attention_forward(
                    vlm_hidden_states=vlm, action_hidden_states=act,
                    layer_idx=i, attention_mask=mask4d,
                    position_ids=position_ids, past_key_values=None,
                    use_cache=False, cache_position=None)
                # VLM-ПОТОК ОТСОЕДИНЯЕТСЯ. Он не видит action-ключей и не
                # содержит обучаемого, так что градиент по нему ровно ноль;
                # но совместное внимание пометило бы его requires_grad, и
                # граф держал бы MLP всех VLM-токенов 12 слоёв.
                vlm = vlm.detach()
            return vlm, act

        # Префикс — дословно как в forward_depth_aligned_rvq: тот же cat,
        # та же маска, те же вызовы. Иначе q0 мог бы разойтись в битах.
        with torch.no_grad():
            bos = model.bos_embedding.expand(batch, n_pos, -1).to(device,
                                                                  dtype)
            empty = torch.empty((batch, 0, bos.shape[-1]), device=device,
                                dtype=dtype)
            act = torch.cat([bos, empty], dim=1)
            mask4d = model._build_joint_attention_mask_blockwise_ar(
                attention_mask=attention_mask,
                vlm_seq_len=vlm_inputs_embeds.shape[1],
                action_seq_len=n_pos, device=device,
                action_key_mask=torch.ones((batch, n_pos), device=device,
                                           dtype=torch.long))
            vlm, act = run_layers(vlm_inputs_embeds, act, 0, Q0_DEPTH)
            q0 = model._joint_depth_logits(act, 0).argmax(dim=-1)
            z0 = model.depth_aligned_book(0)[q0]
            a0 = decode(z0).float()
        if a0.shape[1:] != (self.horizon, 7):
            raise RuntimeError(f"a0 формы {tuple(a0.shape)}")
        if len(layers) != self.levels[-1][2]:
            raise RuntimeError("число слоёв модели изменилось")

        out = dict(q0=q0, a0=a0, actions={}, logits={}, hidden={})
        prev = a0
        lg_prev = grip_logit(a0[:, :self.h_exec, 6])
        out["logit0"] = lg_prev
        for name, lo, hi in self.levels:
            grad = (name == train_level)
            if replace_prev.get(name) is not None:
                prev = replace_prev[name].float().to(device)
                lg_prev = grip_logit(prev[:, :self.h_exec, 6])
            with (torch.enable_grad() if grad else torch.no_grad()):
                mode = fb_mode.get(name, "normal")
                if isinstance(mode, torch.Tensor):
                    # Явный чужой план (из далёкого батча): соседние строки
                    # батча — соседние кадры одного эпизода, и перемешивание
                    # внутри батча почти ничего бы не меняло.
                    if mode.shape != prev.shape:
                        raise ValueError(f"чужой план {tuple(mode.shape)} "
                                         f"против {tuple(prev.shape)}")
                    fb_in = mode.float().to(prev.device)
                elif mode == "shuffle":
                    fb_in = prev.roll(1, dims=0)
                else:
                    fb_in = prev
                with _fp32(device):
                    inj = self.feedback[name](self.feat(fb_in))
                if isinstance(mode, str) and mode == "zero":
                    inj = torch.zeros_like(inj)
                act = act + inj.to(act.dtype)
                with self.active(name):
                    vlm, act = run_layers(vlm, act, lo, hi)
                with _fp32(device):
                    r = self.heads[name](act, self.feat(prev))
                    a_k, lg_k = self.combine(prev, lg_prev, r)
            out["actions"][name] = a_k
            out["logits"][name] = lg_k
            if keep_hidden:
                out["hidden"][name] = act
            if name == stop_after:
                break
            # Следующий уровень видит ФАКТИЧЕСКИЙ план. В фазе 2 он и так
            # без графа: уровень 1 считался под no_grad.
            prev, lg_prev = a_k.detach(), lg_k.detach()
        return out

    def combine(self, prev, lg_prev, r):
        """a_k из предыдущего плана и Δ-логитов r: [B,H_EXEC,7]."""
        h = self.h_exec
        arm = prev[:, :h, :6] + self.delta_scale[:6] * torch.tanh(r[..., :6])
        lg = lg_prev + self.delta_scale[6] * torch.tanh(r[..., 6])
        g = prev[:, :h, 6] + (torch.tanh(lg * 0.5) - torch.tanh(lg_prev * 0.5))
        exec_part = torch.cat([arm, g[..., None]], dim=-1)
        return torch.cat([exec_part, prev[:, h:]], dim=1), lg

    # --- сериализация --------------------------------------------------------
    def export(self) -> dict:
        return dict(kind="k15d_refiner", variant=self.variant,
                    hp=dict(self.hp), d_model=self.d_model,
                    n_pos=self.n_pos, horizon=self.horizon,
                    h_exec=self.h_exec,
                    stats=dict(
                        delta_scale=self.delta_scale.cpu().tolist(),
                        feat_scale=self.feat_scale.cpu().tolist(),
                        loss_scale=self.loss_scale.cpu().tolist()),
                    state={k: v.detach().cpu().clone()
                           for k, v in self.state_dict().items()},
                    state_sha1=state_sha(self))

    @classmethod
    def from_export(cls, obj: dict, *, norm_src, layers, device=None):
        if obj.get("kind") != "k15d_refiner":
            raise ValueError(f"не уточнение K-15d: {obj.get('kind')!r}")
        ref = cls(obj["variant"], norm_src=norm_src, layers=layers,
                  d_model=obj["d_model"], n_pos=obj["n_pos"],
                  horizon=obj["horizon"], h_exec=obj["h_exec"],
                  stats=obj["stats"], hp=obj["hp"])
        if device is not None:
            ref = ref.to(device)
        missing, unexpected = ref.load_state_dict(obj["state"], strict=True)
        if missing or unexpected:
            raise ValueError(f"состояние: нет {missing}, лишние {unexpected}")
        got = state_sha(ref)
        if obj.get("state_sha1") is not None and got != obj["state_sha1"]:
            raise ValueError(f"отпечаток состояния {got}, записан "
                             f"{obj['state_sha1']}")
        ref.set_phase(None)
        return ref

    def count(self, phase=None):
        it = self.named_parameters() if phase is None \
            else self.phase_named_parameters(phase)
        return int(sum(p.numel() for _n, p in it))


@contextmanager
def _fp32(device):
    """Головы и φ считаются в fp32 независимо от внешнего autocast."""
    with torch.autocast(device_type=torch.device(device).type, enabled=False):
        yield


def state_sha(module: nn.Module, names=None) -> str:
    acc = hashlib.sha1()
    for k, v in sorted(module.state_dict().items()):
        if names is not None and k not in names:
            continue
        t = v.detach().cpu().contiguous().reshape(-1)
        acc.update(f"{k}|{tuple(v.shape)}|{v.dtype}".encode())
        acc.update(t.view(torch.uint8).numpy().tobytes())
    return acc.hexdigest()[:12]


# --- статистика и потери ----------------------------------------------------
def residual_stats(a0_exec, act_exec, act_p99, hp=None) -> dict:
    """Фиксированные масштабы по train: [N,H,7] a0 и демонстрации.

    delta_scale[:6] = factor · q-квантиль |остатка| руки — граница поправки;
    delta_scale[6]  = шаг логита схвата, достаточный для полного переворота
                      уверенного ±(1−eps);
    loss_scale      = RMS остатка руки — знаменатель Smooth L1;
    feat_scale      = p99 |действия| по train — нормировка входа φ и голов.
    """
    hp = dict(HP if hp is None else hp)
    r = (act_exec[..., :6] - a0_exec[..., :6]).double().reshape(-1, 6)
    if not bool(torch.isfinite(r).all()):
        raise ValueError("остаток не конечен")
    q = torch.quantile(r.abs().float(), hp["delta_quantile"] / 100.0, dim=0)
    delta = (hp["delta_factor"] * q.double()).clamp_min(1e-4)
    rms = r.pow(2).mean(0).sqrt().clamp_min(1e-4)
    full_flip = 4.0 * math.atanh(1.0 - GRIP_EPS)
    p99 = torch.as_tensor(act_p99, dtype=torch.float64).clamp_min(1e-3)
    return dict(delta_scale=[float(x) for x in delta] + [full_flip],
                feat_scale=[float(x) for x in p99],
                loss_scale=[float(x) for x in rms],
                residual_rms=[float(x) for x in rms],
                residual_quantile=[float(x) for x in q],
                quantile=hp["delta_quantile"], factor=hp["delta_factor"],
                grip_eps=GRIP_EPS, rows=int(a0_exec.shape[0]))


def level_loss(a_k, lg_k, target, loss_scale, h_exec, hp=None):
    """Smooth L1 руки в фиксированных масштабах + BCE схвата.

    Знаменатель — постоянная статистика train, не текущий батч (план §7).
    """
    hp = dict(HP if hp is None else hp)
    t = target[:, :h_exec].float()
    d = (a_k[:, :h_exec, :6] - t[..., :6]) / loss_scale
    arm = F.smooth_l1_loss(d, torch.zeros_like(d),
                           beta=float(hp["smooth_l1_beta"]))
    y = (t[..., 6] > 0).float()
    grip = F.binary_cross_entropy_with_logits(lg_k.float(), y)
    total = arm + float(hp["grip_weight"]) * grip
    return total, dict(arm=arm.detach(), grip=grip.detach())


# --- среда: статистика train и сборка на реальной модели ----------------------
def plan_rows(parts):
    """Уникальные строки части плана в порядке первого появления."""
    import numpy as np
    seen, out = set(), []
    for _po, sel in parts:
        for r in np.asarray(sel, np.int64).tolist():
            if r not in seen:
                seen.add(r)
                out.append(r)
    return np.asarray(out, np.int64)


def decode_q0_rows(ctx, rows, chunk=2048):
    """a0 по каноническим кодам q0 без прохода VLM: [N,T,7] fp32 на CPU.

    Равенство этого a0 и a0 из прохода модели проверяет гейт K-15d: q0
    тот же побитово, а декодер детерминирован и построчен.
    """
    torch_ = ctx.torch
    book0 = ctx.model.depth_aligned_book(0)
    q0 = torch_.as_tensor(ctx.q0_can)
    outs = []
    with torch_.no_grad():
        for i in range(0, len(rows), chunk):
            r = torch_.as_tensor(rows[i:i + chunk])
            z0 = book0[q0[r].to(book0.device)]
            outs.append(ctx.decode_fp32(z0).float().cpu())
    return torch_.cat(outs)


def compute_stats(ctx, h_exec, hp=None):
    """Фиксированные масштабы по ПОЛНОЙ части train плана (не урезанной)."""
    import numpy as np
    rows = plan_rows(ctx.parts_full["train"])
    a0 = decode_q0_rows(ctx, rows)
    act = torch.from_numpy(np.asarray(ctx.ACT[rows], np.float32))
    act = act[:, :a0.shape[1], :7]
    st = residual_stats(a0[:, :h_exec], act[:, :h_exec],
                        np.asarray(ctx.act_p99_dataset, np.float64), hp)
    st["rows_sha1"] = hashlib.sha1(
        np.ascontiguousarray(rows).tobytes()).hexdigest()[:12]
    st["horizon"] = int(a0.shape[1])
    st["h_exec"] = int(h_exec)
    st["sha1"] = stats_sha(st)
    return st


def stats_sha(st) -> str:
    keys = ("delta_scale", "feat_scale", "loss_scale", "rows_sha1",
            "horizon", "h_exec", "quantile", "factor", "grip_eps")
    blob = repr([(k, st.get(k)) for k in keys]).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def make_refiner(ctx, variant, stats, seed, hp=None):
    """Уточнение на реальной модели: размеры берутся из неё, не задаются."""
    model = ctx.model
    torch.manual_seed(int(seed))
    d_model = int(model.action_expert.norm.weight.shape[0])
    ref = DepthRefiner(variant, norm_src=model.action_expert.norm,
                       layers=model.action_expert.layers, d_model=d_model,
                       n_pos=int(model.block_size),
                       horizon=int(stats["horizon"]),
                       h_exec=int(stats["h_exec"]), stats=stats, hp=hp)
    ref = ref.to(ctx.dev)
    ref.set_phase(None)
    n_hooks = ref.attach(model)
    return ref, n_hooks


# --- самопроверка без модели -------------------------------------------------
class _FakeLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.self_attn = nn.Module()
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self.self_attn, n, nn.Linear(d, d))
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(d, 2 * d)
        self.mlp.up_proj = nn.Linear(d, 2 * d)
        self.mlp.down_proj = nn.Linear(2 * d, d)


class _FakeNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        v = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(v + 1e-6) * self.weight).to(x.dtype)


class _FakeModel(nn.Module):
    """Интерфейс, который использует DepthRefiner, на маленьких размерах."""

    def __init__(self, d=16, n_pos=4, n_layers=24, vocab=11, dz=6, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.block_size = n_pos
        self.bos_embedding = nn.Parameter(torch.randn(1, 1, d, generator=g))
        self.action_expert = nn.Module()
        self.action_expert.layers = nn.ModuleList(
            [_FakeLayer(d) for _ in range(n_layers)])
        self.action_expert.norm = _FakeNorm(d)
        self.vlm_proj = nn.Linear(d, d)
        self.fast_head = nn.Linear(d, vocab)
        self.book0 = nn.Parameter(torch.randn(vocab, dz, generator=g))
        with torch.no_grad():
            # Крупная голова: иначе q0 одинаков во всех строках, и тест
            # перемешивания между строками ничего бы не проверял.
            self.fast_head.weight.mul_(8.0)
            self.bos_embedding.mul_(0.1)
        for p in self.parameters():
            p.requires_grad_(False)

    def _build_joint_attention_mask_blockwise_ar(self, **_kw):
        return None

    def _shared_attention_forward(self, *, vlm_hidden_states,
                                  action_hidden_states, layer_idx,
                                  attention_mask, position_ids,
                                  past_key_values, use_cache, cache_position):
        L = self.action_expert.layers[layer_idx]
        v = torch.tanh(self.vlm_proj(vlm_hidden_states))
        # Pre-norm, как в Llama: без него 24 игрушечных слоя с ненулевой
        # LoRA разгоняли активации до inf, и тесты сравнивали NaN.
        a = action_hidden_states
        n = self.action_expert.norm
        an = n(a)
        q = L.self_attn.q_proj(an)
        k = L.self_attn.k_proj(torch.cat([v, an], 1))
        val = L.self_attn.v_proj(torch.cat([v, an], 1))
        w = torch.softmax(q @ k.transpose(1, 2) / 4.0, dim=-1)
        a = a + 0.5 * L.self_attn.o_proj(w @ val)
        an = n(a)
        a = a + 0.5 * L.mlp.down_proj(F.silu(L.mlp.gate_proj(an))
                                      * L.mlp.up_proj(an))
        # VLM-поток возвращается без изменений: повторный tanh по 24 слоям
        # сжимал все строки к одной неподвижной точке.
        return vlm_hidden_states, a

    def _joint_depth_logits(self, h, level):
        assert level == 0
        return self.fast_head(self.action_expert.norm(h))

    def depth_aligned_book(self, level):
        assert level == 0
        return self.book0


def _fake_decoder(dz, horizon, n_pos, seed=1):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(n_pos * dz, horizon * 7, generator=g) * 0.3

    def decode(z):
        # Как настоящий make_action_decoder: вне autocast, в fp32.
        with torch.autocast(device_type=z.device.type, enabled=False):
            return (torch.tanh(z.float().flatten(1) @ W) * 1.02).view(
                z.shape[0], horizon, 7)
    return decode


def _fake_setup(variant, seed=0, horizon=20, h_exec=8):
    torch.manual_seed(seed)
    m = _FakeModel(seed=seed)
    decode = _fake_decoder(m.book0.shape[1], horizon, m.block_size)
    stats = dict(delta_scale=[0.3] * 6 + [4 * math.atanh(1 - GRIP_EPS)],
                 feat_scale=[1.0] * 7, loss_scale=[0.2] * 6)
    ref = DepthRefiner(variant, norm_src=m.action_expert.norm,
                       layers=m.action_expert.layers, d_model=16,
                       n_pos=m.block_size, horizon=horizon, h_exec=h_exec,
                       stats=stats, hp=dict(HP, fb_hidden=8, head_proj=4,
                                            head_hidden=12, lora_rank=2))
    ref.attach(m)
    ref.set_phase(None)
    return m, ref, decode


def _fake_inputs(B=5, S=7, d=16, seed=3):
    g = torch.Generator().manual_seed(seed)
    return dict(vlm_inputs_embeds=torch.randn(B, S, d, generator=g),
                attention_mask=torch.ones(B, S), position_ids=None)


def _perturb(ref, which, scale=0.1, seed=11):
    """Ненулевая искусственная инициализация выбранных выходных матриц."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in ref.named_parameters():
            hit = any(n.startswith(w) for w in which) and (
                n.endswith("out.weight") or n.endswith(".B"))
            if hit:
                p.copy_(torch.randn(p.shape, generator=g) * scale)


def selftest():
    n_tests = 0
    for variant in VARIANTS:
        m, ref, decode = _fake_setup(variant)
        x = _fake_inputs()
        names = ref.level_names()
        # 1. нулевая точка: все уровни побитово равны a0, q0 — эталонному
        with torch.no_grad():
            o = ref.run(m, decode=decode, **x)
            act = m.bos_embedding.expand(5, 4, -1)
            act = torch.cat([act, act[:, :0]], 1)
            vlm = x["vlm_inputs_embeds"]
            for i in range(12):
                vlm, act = m._shared_attention_forward(
                    vlm_hidden_states=vlm, action_hidden_states=act,
                    layer_idx=i, attention_mask=None, position_ids=None,
                    past_key_values=None, use_cache=False,
                    cache_position=None)
            q_ref = m._joint_depth_logits(act, 0).argmax(-1)
        assert torch.equal(o["q0"], q_ref)
        ref.fire_counts.clear()
        with torch.no_grad():
            ref.run(m, decode=decode, **x)
        assert set(ref.fire_counts) == ref.expected_hooks()
        assert set(ref.fire_counts.values()) == {1}
        # ПРЕДУСЛОВИЕ тестов перемешивания: планы строк различаются
        assert not torch.equal(o["a0"], o["a0"].roll(1, dims=0)), \
            "a0 одинаков во всех строках: тест перемешивания был бы пуст"
        for n in names:
            assert torch.equal(o["actions"][n], o["a0"]), (variant, n)
            assert o["actions"][n].shape == (5, 20, 7)
        # 2. белые списки фаз не пересекаются и покрывают всё уточнение
        phases = [p for p, v in PHASE_VARIANT.items() if v == variant]
        seen = set()
        for ph in phases:
            s = {n for n, _p in ref.phase_named_parameters(ph)}
            assert s and not (s & seen)
            seen |= s
        assert seen == {n for n, _p in ref.named_parameters()}
        assert not any(p.requires_grad for p in m.parameters())
        # 3. хуки неактивны вне своего уровня: при ненулевой LoRA проход
        # модели без уточнения не меняется. КОНТРОЛЬ: внутри уровня меняется
        _perturb(ref, ("lora.",))
        with torch.no_grad():
            lin = m.action_expert.layers[13].self_attn.q_proj
            z = torch.randn(2, 16)
            base = lin(z)
            assert torch.equal(base, F.linear(z, lin.weight, lin.bias))
            with ref.active(names[0]):
                assert not torch.allclose(lin(z), base)
        # 4. причинность: LoRA и φ влияют на выход только при ненулевой
        # голове; перемешивание входа φ меняет выход только при ненулевом φ
        _perturb(ref, ("heads.",), seed=12)
        with torch.no_grad():
            o_l = ref.run(m, decode=decode, **x)
            for n in names:
                assert bool(torch.isfinite(o_l["actions"][n]).all()), \
                    "игрушечная модель разошлась: сравнения были бы с NaN"
            for n in names:
                assert not torch.equal(o_l["actions"][n], o_l["a0"])
            for n in names:
                o_s = ref.run(m, decode=decode, fb_mode={n: "shuffle"}, **x)
                assert torch.equal(o_s["actions"][n], o_l["actions"][n]), \
                    "КОНТРОЛЬ: при нулевом φ перемешивание ничего не меняет"
            _perturb(ref, ("feedback.",), seed=13)
            o_f = ref.run(m, decode=decode, **x)
            for n in names:
                o_s = ref.run(m, decode=decode, fb_mode={n: "shuffle"}, **x)
                o_z = ref.run(m, decode=decode, fb_mode={n: "zero"}, **x)
                assert not torch.equal(o_s["actions"][n], o_f["actions"][n])
                assert not torch.equal(o_z["actions"][n], o_f["actions"][n])
            n0 = names[0]
            o_x = ref.run(m, decode=decode,
                          fb_mode={n0: o_f["a0"].roll(1, dims=0)}, **x)
            o_s0 = ref.run(m, decode=decode, fb_mode={n0: "shuffle"}, **x)
            assert torch.equal(o_x["actions"][n0], o_s0["actions"][n0])
            if variant == "h1":
                # a1 действительно доходит до уровня 2 через φ2
                o_z2 = ref.run(m, decode=decode, fb_mode={"2": "zero"}, **x)
                assert torch.equal(o_z2["actions"]["1"], o_f["actions"]["1"])
                tch = o_f["a0"] * 0.5
                o_t = ref.run(m, decode=decode, replace_prev={"2": tch}, **x)
                assert not torch.equal(o_t["actions"]["2"],
                                       o_f["actions"]["2"])
                assert torch.equal(o_t["actions"]["1"], o_f["actions"]["1"])
        # 5. хвост чанка за H_EXEC неподвижен, схват ограничен
        for n in names:
            assert torch.equal(o_f["actions"][n][:, 8:], o_f["a0"][:, 8:])
        with torch.no_grad():
            r_big = torch.full((5, 8, 7), 50.0)
            a_big, _ = ref.combine(o_f["a0"], grip_logit(o_f["a0"][:, :8, 6]),
                                   r_big)
            g0 = o_f["a0"][:, :8, 6]
            off = (g0 - torch.tanh(grip_logit(g0) * 0.5)).abs()
            assert bool((a_big[:, :8, 6].abs() <= 1.0 + off + 1e-6).all())
        n_tests += 5

        # 6. градиенты фаз: только свой белый список; фаза 2 не трогает 1
        for ph in phases:
            lv = PHASES[ph]
            train = ref.set_phase(ph)
            o = ref.run(m, decode=decode, train_level=lv, **x)
            tgt = torch.rand(5, 20, 7) * 2 - 1
            loss, _parts = level_loss(o["actions"][lv], o["logits"][lv], tgt,
                                      ref.loss_scale, 8)
            loss.backward()
            for n, p in ref.named_parameters():
                if n in train:
                    assert p.grad is not None and float(p.grad.abs().sum()) \
                        > 0, (ph, n)
                else:
                    assert p.grad is None, (ph, n)
            assert not any(p.grad is not None for p in m.parameters())
            ref.zero_grad(set_to_none=True)
        # КОНТРОЛЬ вырожденности: при нулевой голове LoRA градиента не
        # получает — значит проверка выше действительно его требует
        m2, ref2, dec2 = _fake_setup(variant, seed=4)
        ph = phases[0]
        ref2.set_phase(ph)
        o = ref2.run(m2, decode=dec2, train_level=PHASES[ph], **x)
        loss, _ = level_loss(o["actions"][PHASES[ph]],
                             o["logits"][PHASES[ph]],
                             torch.rand(5, 20, 7) * 2 - 1,
                             ref2.loss_scale, 8)
        loss.backward()
        lora_g = sum(float(p.grad.abs().sum()) for n, p in
                     ref2.named_parameters() if ".B" in n and p.grad is not None)
        head_g = sum(float(p.grad.abs().sum()) for n, p in
                     ref2.named_parameters()
                     if n.endswith("out.weight") and n.startswith("heads.")
                     and p.grad is not None)
        assert lora_g == 0.0 and head_g > 0.0
        ref2.zero_grad(set_to_none=True)
        ref2.set_phase(None)
        n_tests += 2

        # 7. сериализация: новая копия с другим сидом даёт тот же выход
        with torch.no_grad():
            o1 = ref.run(m, decode=decode, **x)
        obj = ref.export()
        torch.manual_seed(99)
        ref3 = DepthRefiner.from_export(obj, norm_src=m.action_expert.norm,
                                        layers=m.action_expert.layers)
        ref.detach_hooks()
        ref3.attach(m)
        with torch.no_grad():
            o3 = ref3.run(m, decode=decode, **x)
        for n in names:
            assert torch.equal(o1["actions"][n], o3["actions"][n])
        assert state_sha(ref3) == obj["state_sha1"]
        bad = dict(obj, state_sha1="000000000000")
        try:
            DepthRefiner.from_export(bad, norm_src=m.action_expert.norm,
                                     layers=m.action_expert.layers)
            raise AssertionError("подменённый отпечаток прошёл")
        except ValueError:
            pass
        ref3.detach_hooks()
        n_tests += 1

    # 8. потери: Smooth L1 и BCE на ручном примере
    a = torch.zeros(1, 2, 7)
    t = torch.zeros(1, 2, 7)
    t[..., 0] = 2.0
    t[..., 6] = 1.0
    ls = torch.ones(6)
    lg = torch.zeros(1, 2)
    tot, parts = level_loss(a, lg, t, ls, 2)
    want_arm = (2.0 - 0.5) / 6.0
    assert abs(float(parts["arm"]) - want_arm) < 1e-6
    assert abs(float(parts["grip"]) - math.log(2.0)) < 1e-6
    assert abs(float(tot) - want_arm - math.log(2.0)) < 1e-6
    # статистика: границы положительны, масштаб остатка — RMS
    a0e = torch.zeros(50, 8, 7)
    ace = torch.zeros(50, 8, 7)
    ace[..., 1] = 0.1
    st = residual_stats(a0e, ace, [1.0] * 7)
    assert abs(st["loss_scale"][1] - 0.1) < 1e-6
    assert abs(st["delta_scale"][1] - 0.2) < 1e-5
    assert st["delta_scale"][0] == 1e-4
    n_tests += 2
    print(f"самопроверка k15d_depth_refine пройдена: {n_tests} блоков, "
          f"оба варианта, сеть проверена на CPU-torch {torch.__version__}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    print("это модуль; запуск — через k15d_train.py")
    return 0


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    sys.exit(main())
