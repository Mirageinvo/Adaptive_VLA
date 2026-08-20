"""ActionCodec helpers for variable-depth RVQ experiments."""

from __future__ import annotations

import os
import sys
from functools import lru_cache

import numpy as np
import torch


@lru_cache(maxsize=1)
def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_codec(
    model_id: str = "ZibinDong/ActionCodec-Base-RVQft",
    root: str | None = None,
    device: str = "cpu",
):
    """Load ActionCodec from the vendored third_party tree."""

    root = root or os.path.join(_repo_root(), "third_party", "actioncodec")
    if root not in sys.path:
        sys.path.insert(0, root)
    from actioncodec.modeling_actioncodec import ActionCodec

    model = ActionCodec.from_pretrained(model_id).to(device).eval()
    return model


def projected_codebooks(model, device: str | None = None) -> torch.Tensor:
    """Return projected codebooks E[level, vocab, dim]."""

    device = device or next(model.parameters()).device
    vocab = model.vocab_size
    idx = torch.arange(vocab, device=device).unsqueeze(0)
    with torch.no_grad():
        out = [q.out_project(q.decode_code(idx))[0] for q in model.vq.quantizers]
    E = torch.stack(out)
    assert E.shape[0] == model.num_quantizers, E.shape
    return E


def reshape_level_major_codes(flat_codes: torch.Tensor, n_positions: int) -> torch.Tensor:
    """Convert flat level-major codes [B, L*P] into [B, P, L]."""
    if flat_codes.ndim != 2:
        raise ValueError(f"Expected [B, L*P], got {tuple(flat_codes.shape)}")
    bsz, total = flat_codes.shape
    if total % n_positions != 0:
        raise ValueError(f"Cannot split {total} tokens into positions={n_positions}")
    n_levels = total // n_positions
    return flat_codes.view(bsz, n_levels, n_positions).permute(0, 2, 1).contiguous()


def encode_actions(model, actions: torch.Tensor, embodiment_id: int = 0) -> torch.Tensor:
    """Encode actions and return level-major codes [B, P, L]."""

    with torch.no_grad():
        flat = model.encode(actions, embodiment_ids=embodiment_id)
    flat = torch.as_tensor(np.asarray(flat), device=actions.device, dtype=torch.long)
    return reshape_level_major_codes(flat, model.n_tokens_per_quantizer)


def latent_from_depth(E: torch.Tensor, codes: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    """Construct latent sum with per-position depths.

    E: [L, V, D]
    codes: [B, P, L]
    depth: [B, P] with values 1..L
    """

    if codes.ndim != 3:
        raise ValueError(f"Expected [B, P, L] codes, got {tuple(codes.shape)}")
    if depth.shape != codes.shape[:2]:
        raise ValueError(f"Depth shape {tuple(depth.shape)} does not match {tuple(codes.shape[:2])}")
    if E.shape[0] != codes.shape[2]:
        raise ValueError(f"Levels mismatch: E has {E.shape[0]}, codes have {codes.shape[2]}")

    latent = E[0][codes[:, :, 0]]
    for level in range(1, E.shape[0]):
        mask = (depth >= (level + 1)).unsqueeze(-1)
        latent = latent + mask * E[level][codes[:, :, level]]
    return latent


def decode_with_depth(
    model,
    E: torch.Tensor,
    codes: torch.Tensor,
    depth: torch.Tensor,
    embodiment_id: int = 0,
) -> torch.Tensor:
    """Decode actions from full codes plus per-position depths."""

    latent = latent_from_depth(E, codes, depth)
    with torch.no_grad():
        rec, _ = model._decode(latent, embodiment_ids=embodiment_id, durations=None)
    return rec[..., :7]


def native_decode_from_codes(model, codes: torch.Tensor, embodiment_id: int = 0) -> torch.Tensor:
    """Native full-depth decode for comparison tests."""

    flat = codes.permute(0, 2, 1).reshape(codes.shape[0], -1).contiguous()
    with torch.no_grad():
        # Torch 2.4 dtypes do not expose dtype.is_integer used by ActionCodec decode().
        # Passing numpy keeps the official path while avoiding that brittle check.
        rec = model.decode(flat.detach().cpu().numpy(), embodiment_ids=embodiment_id, durations=None)
    if isinstance(rec, tuple):
        rec = rec[0]
    if isinstance(rec, np.ndarray):
        rec = torch.from_numpy(rec).to(codes.device)
    else:
        rec = rec.to(codes.device)
    return rec[..., :7]


def assert_full_depth_matches_native(
    model,
    E: torch.Tensor,
    codes: torch.Tensor,
    embodiment_id: int = 0,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> None:
    """Assert that depth=L decode matches model.decode on full tokens."""

    depth = torch.full(codes.shape[:2], codes.shape[2], device=codes.device, dtype=torch.long)
    rec_manual = decode_with_depth(model, E, codes, depth, embodiment_id=embodiment_id)
    rec_native = native_decode_from_codes(model, codes, embodiment_id=embodiment_id)
    if not torch.allclose(rec_manual, rec_native, atol=atol, rtol=rtol):
        max_abs = float((rec_manual - rec_native).abs().max().item())
        raise AssertionError(f"Full-depth manual decode mismatch (max_abs={max_abs:.3e})")
