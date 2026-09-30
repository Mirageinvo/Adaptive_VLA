#!/usr/bin/env python3
"""K-15: trainable depth-aligned RVQ books over the verified Joint12 draft.

The K-14 implementation remains the executable reference.  This module builds
on it without replacing ``forward_joint_depth_rvq``.  The new method
``forward_depth_aligned_rvq`` differs in one architectural fact: C1 and C2 are
parameters, while C0, q0 and the backbone stay frozen.

Keeping both forwards in one object is deliberate.  Before training, C1/C2
are exact copies of the ActionCodec books and the K-14 q1 state can be loaded
into the shared norms, heads and feedback modules.  The initialization gate
can therefore demand bitwise equality of the old and new q0->q1 paths without
confounding the result with model allocation or weight loading.
"""

from __future__ import annotations

import argparse
import copy
from types import SimpleNamespace
from typing import Iterable

import torch
import torch.nn as nn

try:  # support both ``python experiments/...`` and package imports
    from .depth_rvq_joint12 import (DEFAULT_EXITS,
                                    make_joint_depth_rvq_class)
    from .depth_aligned_tokenizer import hard_straight_through
except ImportError:
    from depth_rvq_joint12 import DEFAULT_EXITS, make_joint_depth_rvq_class
    from depth_aligned_tokenizer import hard_straight_through


TRAINABLE_GROUPS = (
    "depth_aligned_c1",
    "depth_aligned_c2",
    "depth_rvq_norms.0.",
    "depth_rvq_norms.1.",
    "depth_rvq_heads.0.",
    "depth_rvq_heads.1.",
    "depth_rvq_feedback.0.",
    "depth_rvq_feedback.1.",
)


def _matches(name: str, pattern: str) -> bool:
    """Match an exact parameter name or a module prefix ending in a dot."""
    return name.startswith(pattern) if pattern.endswith(".") else name == pattern


def _finite(x: torch.Tensor, name: str) -> None:
    if not torch.isfinite(x).all():
        raise ValueError(f"{name}: contains nan or inf")


def make_depth_aligned_joint12_class(base_cls):
    """Add trainable C1/C2 and a K-15 forward to a Joint12 base class.

    ``base_cls`` is the same base passed to ``make_joint_depth_rvq_class`` in
    K-14.  The returned class retains the legacy forward under its original
    name and exposes the K-15 path as ``forward_depth_aligned_rvq``.
    """

    legacy_cls = make_joint_depth_rvq_class(base_cls)

    class _DepthAlignedJoint12(legacy_cls):

        def init_depth_aligned_rvq(
            self,
            *,
            refine_norm: nn.Module,
            books: torch.Tensor,
            exits: Iterable[int] = DEFAULT_EXITS,
            head_dtype: torch.dtype = torch.float32,
            feedback: bool = True,
            verbose_init: bool = True,
        ):
            if hasattr(self, "depth_aligned_c1"):
                raise RuntimeError("depth-aligned RVQ is already initialized")
            if not feedback:
                raise ValueError(
                    "K-15 requires both draft feedback modules to be built")

            # Build the unchanged K-14 modules and preserve their fixed books
            # as the initialization reference used by the identity gate.
            self.init_joint_depth_rvq(
                refine_norm=refine_norm,
                books=books,
                exits=exits,
                head_dtype=head_dtype,
                feedback=True,
                verbose_init=False,
            )
            ref = self.depth_rvq_books
            if ref.ndim != 3 or ref.shape[0] != 3:
                raise RuntimeError(
                    f"legacy books must be (3,V,D), got {tuple(ref.shape)}")
            _finite(ref, "initial books")

            # C0 remains the legacy buffer.  Separate parameters avoid the
            # unsafe pattern of zeroing one slice of a single Parameter: Adam
            # weight decay or optimizer state could still move that slice.
            self.depth_aligned_c1 = nn.Parameter(ref[1].detach().clone())
            self.depth_aligned_c2 = nn.Parameter(ref[2].detach().clone())
            # АЛИАСИНГ ПРОВЕРЯЕТСЯ, А НЕ ПОДРАЗУМЕВАЕТСЯ. Если C1 или C2
            # окажутся видом на буфер legacy, обучение молча двигало бы
            # эталон, с которым сверяется гейт, и C0 через общее хранилище.
            base_ptr = self.depth_rvq_books.data_ptr()
            for level, parameter in ((1, self.depth_aligned_c1),
                                     (2, self.depth_aligned_c2)):
                if parameter.data_ptr() == base_ptr:
                    raise RuntimeError(
                        f"C{level} делит хранилище с legacy-буфером книг")
                if not parameter.is_leaf or parameter.grad_fn is not None:
                    raise RuntimeError(f"C{level} не является листом графа")
            if self.depth_aligned_c1.data_ptr() == \
                    self.depth_aligned_c2.data_ptr():
                raise RuntimeError("C1 и C2 делят одно хранилище")
            self.depth_aligned_feedback_mask = None

            # Initialization never silently starts training.  The trainer must
            # call configure_depth_aligned_rvq and record the returned names.
            for parameter in self.parameters():
                parameter.requires_grad_(False)
            if verbose_init:
                print(
                    "  depth-aligned RVQ initialized: C0/q0/backbone frozen; "
                    "C1/C2 copy the K-14 books exactly; no trainable stage "
                    "selected"
                )
            return self

        def depth_aligned_book(self, level: int) -> torch.Tensor:
            if not hasattr(self, "depth_aligned_c1"):
                raise RuntimeError("call init_depth_aligned_rvq first")
            level = int(level)
            if level == 0:
                return self.depth_rvq_books[0]
            if level == 1:
                return self.depth_aligned_c1
            if level == 2:
                return self.depth_aligned_c2
            raise ValueError(f"RVQ level must be 0, 1 or 2, got {level}")

        def depth_aligned_books(self) -> tuple[torch.Tensor, ...]:
            return tuple(self.depth_aligned_book(i) for i in range(3))

        def initialization_status(self) -> dict:
            """Return exact initialization checks used by the real gate."""
            if not hasattr(self, "depth_aligned_c1"):
                raise RuntimeError("call init_depth_aligned_rvq first")
            fb0 = self.depth_rvq_feedback[0]
            fb1 = self.depth_rvq_feedback[1]
            bias = fb1.proj.bias
            return {
                "c1_exact": bool(torch.equal(
                    self.depth_aligned_c1.detach(), self.depth_rvq_books[1])),
                "c2_exact": bool(torch.equal(
                    self.depth_aligned_c2.detach(), self.depth_rvq_books[2])),
                "feedback1_weight_zero": bool(
                    torch.count_nonzero(fb1.proj.weight.detach()) == 0),
                "feedback1_bias_zero": bool(
                    bias is None or torch.count_nonzero(bias.detach()) == 0),
                "feedback1_alpha_one": bool(
                    torch.equal(fb1.alpha.detach().float().cpu(),
                                torch.tensor(1.0))),
                # ОБА МОДУЛЯ ПРИ ИНИЦИАЛИЗАЦИИ НУЛЕВЫЕ, И ЭТО НАДО ЗНАТЬ
                # ЯВНО: пока они нулевые, любая проверка «выключение ничего
                # не меняет» тавтологична, и отсоединённый feedback прошёл бы
                # её так же успешно. Гейт обязан отдельно проверить
                # ПРИЧИННОСТЬ на ненулевых весах.
                "feedback0_weight_zero": bool(
                    torch.count_nonzero(fb0.proj.weight.detach()) == 0),
                "feedback0_bias_zero": bool(
                    fb0.proj.bias is None
                    or torch.count_nonzero(fb0.proj.bias.detach()) == 0),
                "books_not_aliased": bool(
                    self.depth_aligned_c1.data_ptr()
                    != self.depth_rvq_books.data_ptr()
                    and self.depth_aligned_c2.data_ptr()
                    != self.depth_rvq_books.data_ptr()),
            }

        def set_depth_aligned_feedback_mask(self, mask=(True, True)) -> tuple:
            if len(mask) != 2:
                raise ValueError(f"feedback mask must have length two: {mask}")
            mask = tuple(bool(x) for x in mask)
            self.depth_aligned_feedback_mask = mask
            # Keep the inherited field synchronized.  The legacy forward uses
            # it, and the identity gate intentionally runs both paths.
            self.depth_rvq_feedback_mask = mask
            return mask

        def configure_depth_aligned_rvq(self, *, verbose: bool = True) -> dict:
            """Freeze everything, then enable exactly the K-15 parameters."""
            if not hasattr(self, "depth_aligned_c1"):
                raise RuntimeError("call init_depth_aligned_rvq first")
            self.set_depth_aligned_feedback_mask((True, True))
            for parameter in self.parameters():
                parameter.requires_grad_(False)
            for name, parameter in self.named_parameters():
                if any(_matches(name, group) for group in TRAINABLE_GROUPS):
                    parameter.requires_grad_(True)

            names = sorted(name for name, parameter in self.named_parameters()
                           if parameter.requires_grad)
            expected = sorted(
                name for name, _parameter in self.named_parameters()
                if any(_matches(name, group) for group in TRAINABLE_GROUPS)
            )
            if names != expected:
                raise RuntimeError(
                    "trainable whitelist mismatch: extra "
                    f"{sorted(set(names) - set(expected))[:5]}, missing "
                    f"{sorted(set(expected) - set(names))[:5]}")
            for group in TRAINABLE_GROUPS:
                if not any(_matches(name, group) for name in names):
                    raise RuntimeError(f"empty trainable group: {group}")

            # The fixed draft and decoder-facing base book are explicit
            # invariants rather than consequences of naming conventions.
            # ВТОРАЯ ЛИНИЯ ОБОРОНЫ НАЗЫВАЕТ ВСЁ, ЧТО ВХОДИТ В ПУТЬ q0.
            # `action_expert.norm` — норма уровня 0: _joint_depth_logits при
            # level == 0 берёт именно её и fast_head. В белый список она не
            # попадает, но список запретов обязан её называть, иначе опечатка
            # в префиксе провела бы её молча.
            forbidden = [
                name for name in names
                if name.startswith("fast_head.")
                or name.startswith("action_expert.norm.")
                or name.startswith("action_expert.layers.")
                or name == "bos_embedding"
            ]
            if forbidden:
                raise RuntimeError(f"frozen draft leaked into optimizer: {forbidden[:5]}")
            if self.depth_rvq_books.requires_grad:
                raise RuntimeError("legacy C0/C1/C2 reference buffer requires grad")

            n_params = sum(parameter.numel() for name, parameter
                           in self.named_parameters() if name in set(names))
            info = {
                "stage": "q1q2",
                "variant": "depth_aligned",
                "groups": list(TRAINABLE_GROUPS),
                "names": names,
                "n_tensors": len(names),
                "n_params": int(n_params),
                "feedback_mask": [True, True],
            }
            if verbose:
                print(
                    f"  K-15 trainable set: {len(names)} tensors, "
                    f"{n_params / 1e6:.3f}M parameters; q0/C0/backbone frozen"
                )
            return info

        def forward_depth_aligned_rvq(
            self,
            *,
            vlm_inputs_embeds: torch.Tensor,
            attention_mask: torch.Tensor,
            position_ids: torch.Tensor,
            mode: str = "full",
            tau: float = 1.0,
        ):
            """One segmented q0->q1->q2 pass with trainable C1/C2.

            The hard value used in the forward pass is always the argmax book
            row.  q1/q2 retain a soft backward path.  No teacher forcing is
            accepted here: K-15 trains and evaluates on its predicted prefix.
            """
            stop = {"fast": 0, "medium": 1, "full": 2}
            if mode not in stop:
                raise ValueError(f"unknown mode {mode}")
            if not float(tau) > 0.0:
                raise ValueError(f"tau must be positive, got {tau}")
            if self.depth_aligned_feedback_mask is None:
                raise RuntimeError(
                    "feedback mask is unset; call configure_depth_aligned_rvq "
                    "or set_depth_aligned_feedback_mask before forward")

            stop_level = stop[mode]
            batch = int(vlm_inputs_embeds.shape[0])
            device, dtype = vlm_inputs_embeds.device, vlm_inputs_embeds.dtype
            n_pos = int(self.block_size)
            bos = self.bos_embedding.expand(batch, n_pos, -1).to(device, dtype)
            empty = torch.empty((batch, 0, bos.shape[-1]),
                                device=device, dtype=dtype)
            action_hidden = torch.cat([bos, empty], dim=1)
            vlm_hidden = vlm_inputs_embeds
            mask4d = self._build_joint_attention_mask_blockwise_ar(
                attention_mask=attention_mask,
                vlm_seq_len=vlm_inputs_embeds.shape[1],
                action_seq_len=n_pos,
                device=device,
                action_key_mask=torch.ones(
                    (batch, n_pos), device=device, dtype=torch.long),
            )

            logits = []
            pred_codes = []
            policy_embeddings = []
            policy_probabilities = []
            cumulative_latents = []
            cumulative = None
            layers_run = 0

            for layer_idx in range(len(self.action_expert.layers)):
                vlm_hidden, action_hidden = self._shared_attention_forward(
                    vlm_hidden_states=vlm_hidden,
                    action_hidden_states=action_hidden,
                    layer_idx=layer_idx,
                    attention_mask=mask4d,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    cache_position=None,
                )
                layers_run += 1
                depth = layer_idx + 1
                if depth not in self.depth_rvq_exits:
                    continue

                level = self.depth_rvq_exits.index(depth)
                level_logits = self._joint_depth_logits(action_hidden, level)
                indices = level_logits.argmax(dim=-1)
                book = self.depth_aligned_book(level)
                if level == 0:
                    embedding = book[indices]
                    probabilities = None
                else:
                    embedding, hard_indices, probabilities = hard_straight_through(
                        level_logits.float(), book, temperature=float(tau))
                    if not torch.equal(indices, hard_indices):
                        raise RuntimeError("straight-through argmax changed indices")

                logits.append(level_logits)
                pred_codes.append(indices)
                policy_embeddings.append(embedding)
                policy_probabilities.append(probabilities)
                cumulative = embedding if cumulative is None else cumulative + embedding
                cumulative_latents.append(cumulative)

                if level >= stop_level:
                    break
                if self.depth_rvq_feedback_built \
                        and self.depth_aligned_feedback_mask[level]:
                    action_hidden = self.depth_rvq_feedback[level](
                        action_hidden, embedding)

            expected_layers = int(self.depth_rvq_exits[stop_level])
            if layers_run != expected_layers:
                raise RuntimeError(
                    f"mode={mode}: ran {layers_run} layers, expected "
                    f"{expected_layers}")
            if len(logits) != stop_level + 1:
                raise RuntimeError(
                    f"mode={mode}: produced {len(logits)} levels, expected "
                    f"{stop_level + 1}")
            return {
                "logits": logits,
                "pred_codes": pred_codes,
                "policy_embeddings": policy_embeddings,
                "policy_probabilities": policy_probabilities,
                "cumulative_latents": cumulative_latents,
                "layers_run": layers_run,
            }

    return _DepthAlignedJoint12


def selftest() -> None:
    """CPU-only architectural checks; the real identity gate uses the VLA."""
    torch.manual_seed(15)
    dim, vocab, latent_dim, positions, n_layers = 8, 11, 5, 3, 24

    class FakeExpert(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(
                [nn.Linear(dim, dim, bias=False) for _ in range(n_layers)])
            self.norm = nn.LayerNorm(dim)

    class FakeBase(nn.Module):
        def __init__(self):
            super().__init__()
            self.action_expert = FakeExpert()
            self.action_lm_head = nn.Linear(dim, vocab)
            self.fast_head = nn.Linear(dim, vocab)
            self.fast_depth = 12
            self.block_size = positions
            self.bos_embedding = nn.Parameter(torch.randn(1, positions, dim))
            self.unrelated = nn.Parameter(torch.randn(2))
            self.config = SimpleNamespace(
                vlm_config=SimpleNamespace(
                    text_config=SimpleNamespace(num_hidden_layers=n_layers)))

        def _build_joint_attention_mask_blockwise_ar(self, **_kwargs):
            return None

        def _shared_attention_forward(
            self, *, vlm_hidden_states, action_hidden_states, layer_idx, **_kwargs
        ):
            action_hidden_states = action_hidden_states + 0.04 * torch.tanh(
                self.action_expert.layers[layer_idx](action_hidden_states))
            return vlm_hidden_states, action_hidden_states

    cls = make_depth_aligned_joint12_class(FakeBase)
    model = cls()
    books = torch.randn(3, vocab, latent_dim)
    model.init_depth_aligned_rvq(
        refine_norm=copy.deepcopy(model.action_expert.norm),
        books=books,
        verbose_init=False,
    )
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert all(model.initialization_status().values()), model.initialization_status()

    # Mimic a nontrivial loaded K-14 q1 checkpoint.  q1 feedback affects h18;
    # feedback1 remains zero so its on/off state must be an exact identity.
    with torch.no_grad():
        nn.init.normal_(model.depth_rvq_heads[0].weight, std=0.2)
        if model.depth_rvq_heads[0].bias is not None:
            nn.init.normal_(model.depth_rvq_heads[0].bias, std=0.1)
        nn.init.normal_(model.depth_rvq_feedback[0].proj.weight, std=0.05)
        nn.init.normal_(model.depth_rvq_feedback[0].proj.bias, std=0.05)

    model.configure_joint_depth_rvq(stage="q1", variant="main", verbose=False)
    model.set_depth_aligned_feedback_mask((True, True))
    x = torch.randn(2, 4, dim)
    old = model.forward_joint_depth_rvq(
        vlm_inputs_embeds=x, attention_mask=None, position_ids=None,
        mode="full")
    new = model.forward_depth_aligned_rvq(
        vlm_inputs_embeds=x, attention_mask=None, position_ids=None,
        mode="full")
    assert old["layers_run"] == new["layers_run"] == 24
    for left, right in zip(old["logits"], new["logits"]):
        assert torch.equal(left, right)
    for left, right in zip(old["pred_codes"], new["pred_codes"]):
        assert torch.equal(left, right)
    # The new ST estimator exposes the exact hard book row.  The legacy ST
    # expression can differ from that row by one ULP, so the meaningful
    # initialization identity is against the actually executed hard codes.
    for level in (0, 1, 2):
        expected = model.depth_aligned_book(level)[new["pred_codes"][level]]
        assert torch.equal(expected, new["policy_embeddings"][level])

    old_z1 = (model.depth_rvq_books[0][old["pred_codes"][0]]
              + model.depth_rvq_books[1][old["pred_codes"][1]])
    assert torch.equal(old_z1, new["cumulative_latents"][1])

    # A zero initialized feedback1 must not change q2 by even one bit.
    model.set_depth_aligned_feedback_mask((True, True))
    fb_on = model.forward_depth_aligned_rvq(
        vlm_inputs_embeds=x, attention_mask=None, position_ids=None,
        mode="full")
    model.set_depth_aligned_feedback_mask((True, False))
    fb_off = model.forward_depth_aligned_rvq(
        vlm_inputs_embeds=x, attention_mask=None, position_ids=None,
        mode="full")
    for left, right in zip(fb_on["logits"], fb_off["logits"]):
        assert torch.equal(left, right)
    for left, right in zip(fb_on["pred_codes"], fb_off["pred_codes"]):
        assert torch.equal(left, right)

    # --- ПРИЧИННОСТЬ: ВЫКЛЮЧЕНИЕ ОБЯЗАНО ЧТО-ТО МЕНЯТЬ -------------------
    # Проверки выше показывают, что НУЛЕВОЙ feedback ничего не меняет. Это
    # тавтология: отсоединённый модуль прошёл бы их так же успешно. Ниже
    # каждый путь проверяется на НЕНУЛЕВЫХ весах, и там выключение обязано
    # изменить соответствующий уровень. Без этих двух проверок гейт не
    # отличает подключённый feedback от неподключённого.
    model.set_depth_aligned_feedback_mask((False, True))
    fb0_off = model.forward_depth_aligned_rvq(
        vlm_inputs_embeds=x, attention_mask=None, position_ids=None,
        mode="full")
    model.set_depth_aligned_feedback_mask((True, True))
    assert not torch.equal(fb_on["logits"][1], fb0_off["logits"][1]), \
        "выключение НЕНУЛЕВОГО feedback0 не изменило q1: путь не подключён"

    with torch.no_grad():
        # Детерминированный узор, а не случайные веса: проба должна быть
        # воспроизводима и не зависеть от состояния генератора.
        weight = model.depth_rvq_feedback[1].proj.weight
        probe = torch.linspace(-0.1, 0.1, weight.numel(),
                               dtype=weight.dtype).reshape(weight.shape)
        saved = weight.detach().clone()
        weight.copy_(probe)
    probe_on = model.forward_depth_aligned_rvq(
        vlm_inputs_embeds=x, attention_mask=None, position_ids=None,
        mode="full")
    model.set_depth_aligned_feedback_mask((True, False))
    probe_off = model.forward_depth_aligned_rvq(
        vlm_inputs_embeds=x, attention_mask=None, position_ids=None,
        mode="full")
    model.set_depth_aligned_feedback_mask((True, True))
    assert not torch.equal(probe_on["logits"][2], probe_off["logits"][2]), \
        "выключение НЕНУЛЕВОГО feedback1 не изменило q2: путь не подключён"
    # уровни 0 и 1 стоят ДО feedback1 и меняться не имеют права
    for level in (0, 1):
        assert torch.equal(probe_on["logits"][level],
                           probe_off["logits"][level]), \
            f"feedback1 повлиял на уровень {level}, который стоит до него"
    with torch.no_grad():
        model.depth_rvq_feedback[1].proj.weight.copy_(saved)
    restored = model.forward_depth_aligned_rvq(
        vlm_inputs_embeds=x, attention_mask=None, position_ids=None,
        mode="full")
    for left, right in zip(fb_on["logits"], restored["logits"]):
        assert torch.equal(left, right), \
            "после восстановления feedback1 состояние не вернулось"

    info = model.configure_depth_aligned_rvq(verbose=False)
    trainable = {name for name, parameter in model.named_parameters()
                 if parameter.requires_grad}
    assert trainable == set(info["names"])
    assert "depth_aligned_c1" in trainable
    assert "depth_aligned_c2" in trainable
    assert not model.fast_head.weight.requires_grad
    assert not model.action_expert.layers[0].weight.requires_grad

    model.zero_grad(set_to_none=True)
    out = model.forward_depth_aligned_rvq(
        vlm_inputs_embeds=x, attention_mask=None, position_ids=None,
        mode="full")
    loss = sum(value.float().square().mean() for value in out["logits"][1:])
    loss = loss + sum(value.float().square().mean()
                      for value in out["cumulative_latents"][1:])
    loss.backward()
    missing = [name for name, parameter in model.named_parameters()
               if parameter.requires_grad and parameter.grad is None]
    nonfinite = [name for name, parameter in model.named_parameters()
                 if parameter.requires_grad and parameter.grad is not None
                 and not torch.isfinite(parameter.grad).all()]
    assert not missing, missing
    assert not nonfinite, nonfinite

    # Moving the new book changes only the new latent path; the legacy book is
    # retained as an immutable identity reference.
    reference_c1 = model.depth_rvq_books[1].clone()
    with torch.no_grad():
        model.depth_aligned_c1.add_(0.125)
    assert torch.equal(model.depth_rvq_books[1], reference_c1)
    moved = model.forward_depth_aligned_rvq(
        vlm_inputs_embeds=x, attention_mask=None, position_ids=None,
        mode="medium")
    assert not torch.equal(moved["cumulative_latents"][1], old_z1)

    print(
        "depth_aligned_joint12 selftest passed: legacy/new initialization "
        "is bitwise identical, zero feedback1 is an identity, C0/q0/backbone "
        "stay frozen, C1/C2 and both depth stages receive finite gradients"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if not args.selftest:
        parser.error("only --selftest is implemented in this module")
    selftest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
