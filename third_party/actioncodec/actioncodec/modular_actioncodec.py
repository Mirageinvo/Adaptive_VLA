import math
from copy import deepcopy
from typing import List, Literal, Optional, Tuple, Union

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from actioncodec.configuration_actioncodec import ActionCodecConfig


def apply_rotary_pos_emb(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    original_dtype = x.dtype

    x = x.to(torch.float32)
    sin = sin.to(torch.float32)
    cos = cos.to(torch.float32)

    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    x_out = torch.empty_like(x)
    x_out[..., 0::2] = rotated_x1
    x_out[..., 1::2] = rotated_x2

    return x_out.to(original_dtype)


def attention_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """

    Args:
        q (torch.Tensor): (*b, h, l, d)
        k (torch.Tensor): (*b, k, s, d)
        v (torch.Tensor): (*b, k, s, d)
        mask (torch.Tensor | None, optional): (*b, l, s), where `True` indicates the element should take part in attention. Defaults to None.
        is_causal (bool, optional): Whether to apply causal mask. Defaults to False.

    Returns:
        torch.Tensor: (*b, h, l, d)
    """
    heads, kv_heads = q.shape[-3], k.shape[-3]
    if heads != kv_heads:
        assert heads % kv_heads == 0, (
            f"q_heads must be divisible by kv_heads, but got {heads} and {kv_heads}"
        )
        heads_per_kv_head = heads // kv_heads
        k, v = map(lambda t: t.repeat_interleave(heads_per_kv_head, dim=1), (k, v))

    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
            mask = mask.expand(mask.shape[0], heads, -1, -1)

    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=is_causal
    )
    return out


class L2Norm(nn.Module):
    def forward(self, x: torch.Tensor):
        return F.normalize(x, p=2, dim=-1)


class Attention(nn.Module):
    """
    Args:
        hidden_size (int): Hidden size of the input tensor.
        num_heads (int): Number of attention heads.
        num_kv_heads (int, optional): Number of key/value heads. Defaults to None.
        qk_norm (Literal["l2", "ln", "none"], optional): Type of normalization to apply to query/key. Defaults to "none".
        bias (bool, optional): Whether to use bias in linear layers. Defaults to False.

    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int = None,
        qk_norm: Literal["l2", "ln", "none"] = "none",
        bias: bool = False,
        zero_init_output: bool = False,
    ):
        super().__init__()
        num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.dim = hidden_size // num_heads
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.dim * num_kv_heads, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.dim * num_kv_heads, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        if qk_norm == "l2":
            self.q_norm = L2Norm()
            self.k_norm = L2Norm()
        elif qk_norm == "ln":
            self.q_norm = nn.LayerNorm(self.dim, elementwise_affine=False)
            self.k_norm = nn.LayerNorm(self.dim, elementwise_affine=False)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        if zero_init_output:
            nn.init.zeros_(self.out_proj.weight)
            if self.out_proj.bias is not None:
                nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor = None,
        mask: torch.Tensor = None,
        rotary_pos_emb: Tuple[torch.Tensor, torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        context = x if context is None else context

        q = self.q_proj(x)
        k, v = self.k_proj(context), self.v_proj(context)

        q = einops.rearrange(q, "b l (h d) -> b h l d", h=self.num_heads)
        k = einops.rearrange(k, "b s (h d) -> b h s d", h=self.num_kv_heads)
        v = einops.rearrange(v, "b s (h d) -> b h s d", h=self.num_kv_heads)

        q, k = self.q_norm(q), self.k_norm(k)

        if rotary_pos_emb is not None:
            q, k = map(lambda t: apply_rotary_pos_emb(t, *rotary_pos_emb), (q, k))

        out = attention_op(q, k, v, mask=mask, is_causal=is_causal)
        out = einops.rearrange(out, "b h l d -> b l (h d)")
        out = self.out_proj(out)

        return out


class PositionalEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        encoding_type: Literal["sincos", "fourier"] = "sincos",
        scale: float = 2.0,
    ):
        super().__init__()
        self.dim = dim
        self.encoding_type = encoding_type

        if encoding_type == "fourier":
            self.register_buffer("freqs", torch.randn(dim // 2) * scale, persistent=True)
        elif encoding_type == "sincos":
            pass
        else:
            raise ValueError(
                f"encoding_type must be 'sincos' or 'fourier', but got {encoding_type}"
            )

    def _create_sincos_emb(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        position = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
            * -(math.log(10000.0) / self.dim)
        )

        pos_emb = torch.zeros(seq_len, self.dim, device=device, dtype=dtype)
        pos_emb[:, 0::2] = torch.sin(position * div_term).to(dtype)
        pos_emb[:, 1::2] = torch.cos(position * div_term).to(dtype)

        return pos_emb

    def _create_fourier_emb(
        self, timestamps: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        # Ensure freqs is on the correct device
        freqs = self.freqs.to(device)
        pos_emb = torch.einsum("b t, d -> b t d", timestamps, 2 * np.pi * freqs).to(
            device, torch.float32
        )
        pos_emb = torch.cat([pos_emb.cos(), pos_emb.sin()], dim=-1).to(dtype)
        return pos_emb

    def forward(
        self,
        x: torch.Tensor,
        freq: Optional[Union[float, torch.Tensor]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        b, t = x.shape[0], x.shape[1]
        device = x.device

        if self.encoding_type == "sincos":
            pos_emb = self._create_sincos_emb(t, device, dtype)
            pos_emb = pos_emb.unsqueeze(0).expand(b, -1, -1)
            return pos_emb * 0.1

        elif self.encoding_type == "fourier":
            if freq is None:
                raise ValueError(
                    "freq must be provided when encoding_type is 'fourier'. Please provide the sequence frequency."
                )
            if isinstance(freq, float):
                freq = torch.tensor(freq, dtype=dtype, device=device)[None].expand(b)
            timestamps = torch.einsum(
                "t, b -> b t", torch.arange(t, dtype=dtype, device=device), 1 / freq
            )
            pos_emb = self._create_fourier_emb(timestamps, device, dtype)
            return pos_emb * 0.1
        else:
            raise ValueError(f"Unknown encoding_type: {self.encoding_type}")


class SinusoidalPositionalEmbedding(PositionalEmbedding):
    def __init__(self, dim: int):
        super().__init__(dim=dim, encoding_type="sincos")

    def forward(self, x: torch.Tensor, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        return super().forward(x, freq=None)


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act_fn = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down_proj = self.down_proj(self.act_fn(self.up_proj(x)))
        return down_proj


class LayerScale(nn.Module):
    def __init__(self, dim, init_val=1e-2):
        super().__init__()
        self.scale = nn.Parameter(torch.full([dim], init_val))

    def forward(self, x):
        return x * self.scale


class PerceiverTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        qk_norm: str = "ln",
        layer_scale: bool = True,
        zero_init_output: bool = False,
        add_self_attn: bool = False,
        add_causal_mask: bool = False,
    ):
        super().__init__()
        self.add_self_attn = add_self_attn
        self.add_causal_mask = add_causal_mask

        self.norm1 = nn.LayerNorm(dim, eps=1e-2)
        self.cross_attn = Attention(
            hidden_size=dim,
            num_heads=num_heads,
            qk_norm=qk_norm,
            bias=False,
            zero_init_output=zero_init_output,
        )

        if add_self_attn:
            self.norm_self_attn = nn.LayerNorm(dim, eps=1e-2)
            self.self_attn = Attention(
                hidden_size=dim,
                num_heads=num_heads,
                qk_norm=qk_norm,
                bias=False,
                zero_init_output=zero_init_output,
            )
        else:
            self.self_attn = None

        self.norm2 = nn.LayerNorm(dim, eps=1e-2)
        self.mlp = FeedForward(hidden_size=dim, intermediate_size=int(mlp_ratio * dim), bias=True)
        self.dropout = nn.Dropout(dropout)

        self.attn_scale = LayerScale(dim) if layer_scale else nn.Identity()
        self.mlp_scale = LayerScale(dim) if layer_scale else nn.Identity()

        if zero_init_output:
            nn.init.zeros_(self.mlp.down_proj.weight)
            if self.mlp.down_proj.bias is not None:
                nn.init.zeros_(self.mlp.down_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.cross_attn(
            x=x, context=context, mask=context_mask, rotary_pos_emb=rotary_pos_emb, is_causal=False
        )
        x = self.dropout(x)
        x = self.attn_scale(x)
        x = x + residual

        if self.add_self_attn:
            residual = x
            x = self.norm_self_attn(x)
            x = self.self_attn(
                x=x,
                context=None,
                mask=None,
                rotary_pos_emb=rotary_pos_emb,
                is_causal=self.add_causal_mask,
            )
            x = self.dropout(x)
            x = self.attn_scale(x)
            x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.dropout(x)
        x = self.mlp_scale(x)
        x = x + residual

        return x


class EmbodimentEmbedding(nn.Module):
    def __init__(self, embodiment_config: dict, out_len: int, out_dim: int) -> None:
        super().__init__()
        self.out_len, self.out_dim = out_len, out_dim

        self.embodiment_config = embodiment_config
        self.num_embodiments = len(self.embodiment_config)

        self.embedding = nn.Embedding(self.num_embodiments, out_dim * out_len)

    @torch.no_grad()
    def expand_embodiment(self, embodiment_config: dict):
        for k in embodiment_config.keys():
            assert k not in self.embodiment_config.keys()
        self.embodiment_config.update(embodiment_config)
        self.num_embodiments = len(self.embodiment_config)

        extra_embodiments = len(embodiment_config)

        old_weights = torch.clone(self.embedding.weight)
        self.embedding = nn.Embedding(self.num_embodiments, self.out_dim * self.out_len)
        self.embedding.weight.data[:-extra_embodiments] = old_weights
        return self

    def keys(self) -> List[str]:
        return list(self.embodiment_config.keys())

    def ids_to_keys(self, ids: torch.Tensor) -> List[str]:
        return [self.keys()[i] for i in ids]

    def keys_to_ids(self, keys: List[str]) -> torch.Tensor:
        return torch.tensor([self.keys().index(k) for k in keys])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.rearrange(self.embedding(x), "b (l d) -> b l d", d=self.out_dim)


class PerceiverEncoder(nn.Module):
    def __init__(self, config: ActionCodecConfig):
        super().__init__()
        self.config = config
        self.embodiment_config = deepcopy(config.embodiment_config)

        out_len = int(config.n_tokens // config.n_quantizers)
        dim = config.encoder_dim

        _action_dim, _freq, _duration = list(), list(), list()
        for k, v in self.embodiment_config.items():
            _action_dim.append(v["action_dim"])
            _freq.append(v["freq"])
            _duration.append(v["duration"])
        self.register_buffer("_action_dim", torch.tensor(_action_dim), persistent=False)
        self.register_buffer("_freq", torch.tensor(_freq), persistent=False)
        self.register_buffer("_duration", torch.tensor(_duration), persistent=False)

        self.max_action_dim = max(v["action_dim"] for v in self.embodiment_config.values())
        self.input_proj = nn.Linear(self.max_action_dim, dim)

        self.cls_tokens = EmbodimentEmbedding(self.embodiment_config, out_len, dim)

        self.pos_emb_q = PositionalEmbedding(dim, encoding_type="sincos")
        self.pos_emb_kv = PositionalEmbedding(dim, encoding_type=config.encoder_pos_encoding_type)

        self.layers = nn.ModuleList(
            [
                PerceiverTransformerBlock(
                    dim=dim,
                    num_heads=config.encoder_n_heads,
                    add_self_attn=config.encoder_add_self_attn,
                    add_causal_mask=config.encoder_add_causal_mask,
                )
                for _ in range(config.encoder_n_layers)
            ]
        )

        self.output_proj = nn.Linear(dim, config.z_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.input_proj.weight, std=0.02)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)
        nn.init.trunc_normal_(self.output_proj.weight, std=0.02)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)

        nn.init.trunc_normal_(self.cls_tokens.embedding.weight, std=0.02)

    @torch.no_grad()
    def expand_embodiment(self, embodiment_config: dict):
        self.cls_tokens.expand_embodiment(embodiment_config)
        self.embodiment_config = self.cls_tokens.embodiment_config
        _action_dim, _freq, _duration = list(), list(), list()
        for k, v in self.embodiment_config.items():
            _action_dim.append(v["action_dim"])
            _freq.append(v["freq"])
            _duration.append(v["duration"])
        self._action_dim = torch.tensor(_action_dim)
        self._freq = torch.tensor(_freq)
        self._duration = torch.tensor(_duration)

        max_action_dim = max(v["action_dim"] for v in self.embodiment_config.values())
        if max_action_dim > self.max_action_dim:
            old_weights = torch.clone(self.input_proj.weight)
            old_bias = torch.clone(self.input_proj.bias)
            self.input_proj = nn.Linear(max_action_dim, self.config.encoder_dim)
            self.input_proj.weight.data[:, : self.max_action_dim] = old_weights
            self.input_proj.bias.data = old_bias
            self.max_action_dim = max_action_dim

        return self

    def forward(
        self,
        x: torch.Tensor,
        embodiment_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode action sequences into latent representations.

        Args:
            x (torch.Tensor): Action sequences to encode. Shape: (b, seq_len, max_action_dim).
                Assumes that the action dimension is zero-padded to the max action dimension.
                `seq_len` is supposed to be `int(duration * freq)` for each embodiment and padded to the max sequence length.
            embodiment_ids (torch.Tensor | int): Embodiment IDs. Shape: (b,).
                If int, the same embodiment ID is repeated for all sequences in the batch.
                It specifies the embodiment to encode.
            padding_mask (Optional[torch.Tensor], optional): Padding mask, where `False` values indicate padding. Shape: (b, seq_len). Defaults to None.
                It is used to mask the padding tokens on `seq_len` dimension.

        Returns:
            torch.Tensor: Encoded latent representations. Shape: (b, n_tokens_per_quantizer, z_dim).
        """
        b, seq_len, _ = x.shape

        x = self.input_proj(x)

        if isinstance(embodiment_ids, int):
            embodiment_ids = torch.tensor(
                [embodiment_ids], dtype=torch.long, device=x.device
            ).repeat(b)

        cls_tokens = self.cls_tokens(embodiment_ids)

        freqs = self._freq[embodiment_ids].to(x.device, x.dtype)

        pos_emb_q = self.pos_emb_q(cls_tokens)
        pos_emb_kv = self.pos_emb_kv(x, freqs)

        cls_tokens = cls_tokens + pos_emb_q
        x = x + pos_emb_kv

        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).expand(-1, cls_tokens.shape[1], -1)

        for layer in self.layers:
            cls_tokens = layer(x=cls_tokens, context=x, context_mask=padding_mask)

        return self.output_proj(cls_tokens)


class PerceiverDecoder(nn.Module):
    def __init__(self, config: ActionCodecConfig):
        super().__init__()
        self.config = config
        self.embodiment_config = deepcopy(config.embodiment_config)

        dim = config.decoder_dim

        _action_dim, _freq, _duration = list(), list(), list()
        for k, v in self.embodiment_config.items():
            _action_dim.append(v["action_dim"])
            _freq.append(v["freq"])
            _duration.append(v["duration"])
        self.register_buffer("_action_dim", torch.tensor(_action_dim), persistent=False)
        self.register_buffer("_freq", torch.tensor(_freq), persistent=False)
        self.register_buffer("_duration", torch.tensor(_duration), persistent=False)

        self.max_action_dim = max(v["action_dim"] for v in self.embodiment_config.values())
        self.input_proj = nn.Linear(config.z_dim, dim)

        self.cls_tokens = EmbodimentEmbedding(self.embodiment_config, config.decoder_cls_size, dim)

        self.pos_emb_q = PositionalEmbedding(dim, encoding_type=config.decoder_pos_encoding_type)
        self.pos_emb_kv = PositionalEmbedding(dim, encoding_type="sincos")

        self.layers = nn.ModuleList(
            [
                PerceiverTransformerBlock(
                    dim=dim,
                    num_heads=config.decoder_n_heads,
                    add_self_attn=config.decoder_add_self_attn,
                    add_causal_mask=config.decoder_add_causal_mask,
                )
                for _ in range(config.decoder_n_layers)
            ]
        )

        self.output_proj = nn.Linear(dim, self.max_action_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.input_proj.weight, std=0.02)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)
        nn.init.trunc_normal_(self.output_proj.weight, std=0.02)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)
        nn.init.trunc_normal_(self.cls_tokens.embedding.weight, std=0.02)

    @torch.no_grad()
    def expand_embodiment(self, embodiment_config: dict):
        self.cls_tokens.expand_embodiment(embodiment_config)
        self.embodiment_config = self.cls_tokens.embodiment_config

        _action_dim, _freq, _duration = list(), list(), list()
        for k, v in self.embodiment_config.items():
            _action_dim.append(v["action_dim"])
            _freq.append(v["freq"])
            _duration.append(v["duration"])
        self._action_dim = torch.tensor(_action_dim)
        self._freq = torch.tensor(_freq)
        self._duration = torch.tensor(_duration)

        max_action_dim = max(v["action_dim"] for v in self.embodiment_config.values())

        if max_action_dim > self.max_action_dim:
            old_weights = torch.clone(self.output_proj.weight)
            old_bias = torch.clone(self.output_proj.bias)

            self.output_proj = nn.Linear(self.config.decoder_dim, max_action_dim)

            self.output_proj.weight.data[: self.max_action_dim, :] = old_weights
            self.output_proj.bias.data[: self.max_action_dim] = old_bias

            self.max_action_dim = max_action_dim

        return self

    def forward(
        self, x: torch.Tensor, embodiment_ids: torch.Tensor, durations: torch.Tensor = None
    ) -> torch.Tensor:
        """Decode latent representations into action sequences.

        Args:
            x (torch.Tensor): Latent representations to decode. Shape: (b, n_tokens_per_quantizer, z_dim).
            embodiment_ids (torch.Tensor | int): Embodiment IDs. Shape: (b,).
                If int, the same embodiment ID is repeated for all sequences in the batch.
                It specifies the embodiment to decode.
            durations (torch.Tensor | None, optional): Duration of each action sequence. Shape: (b,).
                If `None`, the duration is inferred from the default values in `embodiment_config`.

        Returns:
            torch.Tensor: Decoded action sequences. Shape: (b, seq_len, max_action_dim).
                Assumes that the action dimension is zero-padded to the max action dimension.
                `seq_len` is supposed to be `int(duration * freq)` for each embodiment and padded to the max sequence length.
        """
        b, seq_len, _ = x.shape
        x = self.input_proj(x)

        if isinstance(embodiment_ids, int):
            embodiment_ids = torch.tensor(
                [embodiment_ids], dtype=torch.long, device=x.device
            ).repeat(b)

        cls_tokens = self.cls_tokens(embodiment_ids)

        freqs = self._freq[embodiment_ids]
        if freqs.device != x.device:
            freqs = freqs.to(x.device)

        durations = self._duration[embodiment_ids] if durations is None else durations
        if isinstance(durations, torch.Tensor) and durations.device != x.device:
            durations = durations.to(x.device)

        action_horizons = (durations * freqs).long()
        max_horizon = action_horizons.max().item()
        padding_mask = torch.arange(max_horizon, device=x.device).expand(
            b, -1
        ) < action_horizons.unsqueeze(1)

        if self.config.decoder_cls_size == 1:
            cls_tokens = cls_tokens.repeat(1, max_horizon, 1)

        pos_emb_q = self.pos_emb_q(cls_tokens, freqs)
        pos_emb_kv = self.pos_emb_kv(x)

        cls_tokens = cls_tokens + pos_emb_q
        x = x + pos_emb_kv

        for layer in self.layers:
            cls_tokens = layer(x=cls_tokens, context=x)

        output = self.output_proj(cls_tokens)

        return output, padding_mask


if __name__ == "__main__":
    # ------------------------------------------
    # 1. Initialization
    # ------------------------------------------
    print("=== Test 1: Initialization ===")

    # Define initial config with two smaller robots
    initial_embodiment_config = {
        "robot_small_7d": {
            "action_dim": 7,
            "freq": 20,
            "duration": 1,
            "description": "Original Robot",
        },
        "robot_tiny_3d": {"action_dim": 3, "freq": 10, "duration": 2, "description": "Tiny Robot"},
    }

    config = ActionCodecConfig(embodiment_config=initial_embodiment_config)

    # Set seed for reproducibility
    torch.manual_seed(42)

    encoder = PerceiverEncoder(config)
    decoder = PerceiverDecoder(config)

    encoder.eval()
    decoder.eval()
    print("✅ Models initialized successfully.")

    # ------------------------------------------
    # 2. Baseline Inference (Before Expansion)
    # ------------------------------------------
    print("\n=== Test 2: Baseline Inference (Before Expansion) ===")

    # Simulate Robot 1 (7-dim) data
    # Max action dim currently is 7.
    batch_size = 1
    seq_len = 20  # 20Hz * 1s

    # Input: (1, 20, 7)
    input_action_v0 = torch.randn(batch_size, seq_len, 7)
    emb_id_v0 = torch.tensor([0], dtype=torch.long)  # ID 0 -> robot_small_7d

    with torch.no_grad():
        z_ref = encoder(input_action_v0, emb_id_v0)
        rec_action_ref, _ = decoder(z_ref, emb_id_v0)

    print(f"Reference Latent Shape: {z_ref.shape}")
    print(f"Reference Recon Shape: {rec_action_ref.shape}")

    # ------------------------------------------
    # 3. Model Expansion (Add New Embodiment)
    # ------------------------------------------
    print("\n=== Test 3: Model Expansion ===")

    # Add a larger robot: 10-dim, high frequency
    new_embodiment_config = {
        "robot_large_10d": {
            "action_dim": 10,
            "freq": 30,
            "duration": 1,
            "description": "New Large Robot",
        }
    }

    print(f"Expanding from Max Dim {encoder.max_action_dim} to 10...")
    encoder.expand_embodiment(new_embodiment_config)
    decoder.expand_embodiment(new_embodiment_config)

    # Verify buffer updates
    assert encoder._action_dim[-1] == 10
    assert encoder.max_action_dim == 10
    assert decoder.max_action_dim == 10
    print(f"✅ Expansion successful. New Encoder Input Dim: {encoder.input_proj.weight.shape[1]}")
    print(f"✅ New Decoder Output Dim: {decoder.output_proj.weight.shape[0]}")

    # ------------------------------------------
    # 4. Encoder Invariance Check
    # ------------------------------------------
    print("\n=== Test 4: Encoder Invariance Check ===")

    # Pad old data (7 dims) to new max dim (10 dims) with ZEROS.
    input_action_padded = torch.zeros(batch_size, seq_len, 10)
    input_action_padded[:, :, :7] = input_action_v0

    with torch.no_grad():
        z_new = encoder(input_action_padded, emb_id_v0)

    # Compare latents
    diff_z = (z_ref - z_new).abs().max().item()
    print(f"Latent Difference (Max Abs): {diff_z:.8f}")

    if diff_z < 1e-6:
        print("✅ PASS: Encoder produces identical latents for old data.")
    else:
        print("❌ FAIL: Encoder outputs changed after expansion!")

    # ------------------------------------------
    # 5. Decoder Invariance Check
    # ------------------------------------------
    print("\n=== Test 5: Decoder Invariance Check ===")

    with torch.no_grad():
        # Feed old latent to expanded decoder
        rec_action_new_full, _ = decoder(z_ref, emb_id_v0)

    # Output shape should be (1, 20, 10)
    print(f"Expanded Decoder Output Shape: {rec_action_new_full.shape}")

    # Slice first 7 dims, should match reference
    rec_action_new_sliced = rec_action_new_full[:, :, :7]

    diff_rec = (rec_action_ref - rec_action_new_sliced).abs().max().item()
    print(f"Reconstruction Difference (Max Abs on valid dims): {diff_rec:.8f}")

    if diff_rec < 1e-6:
        print("✅ PASS: Decoder produces identical action values for valid dimensions.")
    else:
        print("❌ FAIL: Decoder outputs changed!")

    # Check phantom dimensions (7-9)
    # For old embodiment, these are driven by random weights and should be random
    new_dims_mean = rec_action_new_full[:, :, 7:].abs().mean().item()
    print(f"Values in new phantom dimensions (should be random garbage): {new_dims_mean:.4f}")

    # ------------------------------------------
    # 6. New Embodiment Inference
    # ------------------------------------------
    print("\n=== Test 6: New Embodiment Inference ===")

    # ID 2 -> robot_large_10d
    emb_id_new = torch.tensor([2], dtype=torch.long)
    seq_len_new = 30  # 30Hz * 1s

    input_action_new = torch.randn(1, seq_len_new, 10)

    with torch.no_grad():
        z_large = encoder(input_action_new, emb_id_new)
        rec_large, mask_large = decoder(z_large, emb_id_new)

    print(f"New Embodiment Output Shape: {rec_large.shape}")

    if rec_large.shape == (1, 30, 10):
        print("✅ PASS: New embodiment handled correctly with full dimensions.")
    else:
        print(f"❌ FAIL: Expected (1, 30, 10), got {rec_large.shape}")

    # ------------------------------------------
    # 7. Mixed Batch Processing (Masking)
    # ------------------------------------------
    print("\n=== Test 7: Mixed Batch Processing ===")

    # Batch size 2: [Robot 0 (20Hz, 7dim), Robot 2 (30Hz, 10dim)]
    mixed_emb_ids = torch.tensor([0, 2], dtype=torch.long)

    # Max seq len is 30. Max action dim is 10.
    batch_input = torch.zeros(2, 30, 10)

    # Fill data
    # Batch 0: Length 20, Dim 7 valid
    batch_input[0, :20, :7] = torch.randn(20, 7)
    # Batch 1: Length 30, Dim 10 valid
    batch_input[1, :30, :10] = torch.randn(30, 10)

    # Encoder Mask: True = Valid
    enc_padding_mask = torch.zeros(2, 30, dtype=torch.bool)
    enc_padding_mask[0, :20] = True
    enc_padding_mask[1, :30] = True

    print("Running mixed batch...")
    with torch.no_grad():
        z_mixed = encoder(batch_input, mixed_emb_ids, padding_mask=enc_padding_mask)
        rec_mixed, dec_padding_mask = decoder(z_mixed, mixed_emb_ids)

    print(f"Mixed Reconstruction Shape: {rec_mixed.shape}")  # Should be (2, 30, 10)

    # Verify Decoder Generated Mask
    valid_len_0 = dec_padding_mask[0].sum().item()
    valid_len_1 = dec_padding_mask[1].sum().item()

    print(f"Decoder Mask Valid Lengths: Batch 0={valid_len_0}, Batch 1={valid_len_1}")

    if valid_len_0 == 20 and valid_len_1 == 30:
        print("✅ PASS: Decoder correctly generated masks based on frequency and duration.")
    else:
        print("❌ FAIL: Decoder masks are incorrect.")

    print("\n✨ All Tests Completed ✨")
