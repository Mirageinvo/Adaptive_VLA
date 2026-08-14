from typing import List, Tuple, Union

import einops
import numpy as np
import torch
from transformers import AutoModel, PreTrainedModel
from vector_quantize_pytorch import VectorQuantize

from actioncodec.configuration_actioncodec import ActionCodecConfig
from actioncodec.modular_actioncodec import PerceiverDecoder, PerceiverEncoder
from actioncodec.rvq import ResidualVectorQuantize


def trim_trailing_zeros(arr: np.ndarray) -> List[np.ndarray]:
    if arr.shape[0] == 0:
        return []

    b, n = arr.shape

    is_nonzero = arr != 0
    flipped_mask = np.flip(is_nonzero, axis=1)
    last_nonzero_indices = n - 1 - np.argmax(flipped_mask, axis=1)
    any_nonzero_in_row = is_nonzero.any(axis=1)
    new_lengths = (last_nonzero_indices + 1) * any_nonzero_in_row
    result = [arr[i, :length].tolist() for i, length in enumerate(new_lengths)]

    return result


class ActionCodec(PreTrainedModel):
    """ActionCodec: A neural codec for encoding and decoding robot action sequences.

    This model uses a Perceiver-based encoder-decoder architecture with vector quantization
    to convert continuous action sequences into discrete token sequences. It supports
    multiple robot embodiments with different action dimensions and control frequencies.

    The model supports two vector quantization types:
    - VQ (Vector Quantization): Single quantizer
    - RVQ (Residual Vector Quantization): Multiple quantizers for hierarchical encoding

    Key features:
    - Multi-embodiment support: Handle different robots with varying action dimensions
    - Dynamic expansion: Add new robot configurations without retraining
    - Flexible input/output: Support numpy arrays and torch tensors
    """

    config_class = ActionCodecConfig

    def __init__(self, config: ActionCodecConfig):
        """Initialize the ActionCodec model.

        Args:
            config (ActionCodecConfig): Model configuration containing hyperparameters
                and embodiment configurations.

        Raises:
            ValueError: If configuration parameters are invalid.
            NotImplementedError: If the specified VQ type is not supported.
        """
        super().__init__(config)

        # Validate configuration
        if config.n_tokens % config.n_quantizers != 0:
            raise ValueError(
                f"n_tokens ({config.n_tokens}) must be divisible by n_quantizers ({config.n_quantizers})"
            )

        if config.n_quantizers < 1:
            raise ValueError(f"n_quantizers must be at least 1, got {config.n_quantizers}")

        if config.vq_codebook_size < 1:
            raise ValueError(f"vq_codebook_size must be at least 1, got {config.vq_codebook_size}")

        if config.z_dim < 1:
            raise ValueError(f"z_dim must be at least 1, got {config.z_dim}")

        if not isinstance(config.embodiment_config, dict) or len(config.embodiment_config) == 0:
            raise ValueError(
                "embodiment_config must be a non-empty dictionary mapping embodiment names to configurations"
            )

        self.default_embodiment_id = 0

        # Initialize encoder and decoder
        self.encoder = PerceiverEncoder(config)
        self.decoder = PerceiverDecoder(config)

        # Initialize vector quantizer based on type
        if config.vq_type == "vq":
            if config.n_quantizers != 1:
                raise ValueError(
                    f"VQ type requires n_quantizers=1, got {config.n_quantizers}. Use RVQ type for multiple quantizers."
                )
            self.vq = VectorQuantize(
                dim=config.z_dim,
                codebook_size=config.vq_codebook_size,
                commitment_weight=config.vq_commitment_weight,
                decay=config.vq_decay,
                kmeans_init=config.vq_kmeans_init,
                threshold_ema_dead_code=config.vq_threshold_ema_dead_code,
                rotation_trick=False,
                straight_through=True,
            )
        elif config.vq_type == "rvq":
            if config.n_quantizers < 2:
                raise ValueError(
                    f"RVQ type requires n_quantizers >= 2, got {config.n_quantizers}. Use VQ type for single quantizer."
                )
            self.vq = ResidualVectorQuantize(
                dim=config.z_dim,
                n_codebooks=config.n_quantizers,
                codebook_size=config.vq_codebook_size,
                codebook_dim=config.z_dim,
                quantizer_dropout=config.vq_quantizer_dropout,
                commitment=config.vq_commitment_weight,
            )
        else:
            raise NotImplementedError(
                f"VQ type '{config.vq_type}' not implemented. Supported types: 'vq', 'rvq'"
            )

        # Store quantization-related attributes
        self.vocab_size = config.vq_codebook_size
        self.num_quantizers = config.n_quantizers
        self.n_tokens_per_quantizer = config.n_tokens // config.n_quantizers

    def expand_embodiment(self, embodiment_config: dict):
        """Dynamically expand the model to support new robot embodiments.

        This method allows adding new robot configurations to the codec without retraining
        the entire model. It updates the encoder and decoder to handle the new action dimensions
        and frequencies while preserving existing functionality for previously configured robots.

        Args:
            embodiment_config (dict): Dictionary mapping embodiment names to their configurations.
                Each configuration should be a dict with keys:
                - "action_dim" (int): Action dimensionality for this embodiment.
                - "freq" (float): Control frequency in Hz.
                - "duration" (float): Default action sequence duration in seconds.
                - "description" (str, optional): Human-readable description.

                Example:
                    {
                        "robot_B": {
                            "action_dim": 10,
                            "freq": 20,
                            "duration": 1.0,
                            "description": "10-dim robot at 20Hz"
                        }
                    }

        Returns:
            ActionCodec: Returns self for method chaining.

        Note:
            - New embodiment keys must not already exist in the current configuration.
            - The model will automatically update max_action_dim if the new embodiment
              has a larger action dimension.
            - Existing embodiments will continue to work with their original configurations.
        """
        if not isinstance(embodiment_config, dict):
            raise TypeError(f"embodiment_config must be a dict, got {type(embodiment_config)}")
        if len(embodiment_config) == 0:
            raise ValueError("embodiment_config cannot be empty")

        # Check for duplicate keys
        overlapping_keys = set(embodiment_config.keys()) & set(self.config.embodiment_config.keys())
        if overlapping_keys:
            raise ValueError(
                f"The following embodiment keys already exist and cannot be redefined: {overlapping_keys}"
            )

        self.encoder.expand_embodiment(embodiment_config)
        self.decoder.expand_embodiment(embodiment_config)
        self.config.embodiment_config.update(embodiment_config)
        return self

    def _encode(
        self,
        x: torch.Tensor,
        embodiment_ids: torch.Tensor = None,
        padding_mask: torch.Tensor = None,
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
        embodiment_ids = (
            embodiment_ids if embodiment_ids is not None else self.default_embodiment_id
        )
        z_e = self.encoder(x, embodiment_ids, padding_mask)
        return z_e

    def _quantize(
        self, z_e: torch.Tensor, return_perplexity: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Union[float, List[float]], torch.Tensor]:
        """Quantize encoded representations using vector quantization.

        Args:
            z_e (torch.Tensor): Encoded latent representations to quantize.
                Shape: (b, n_tokens_per_quantizer, z_dim).
            return_perplexity (bool, optional): Whether to compute and return perplexity.
                Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Union[float, List[float]], torch.Tensor]:
                A tuple containing:
                - z_q (torch.Tensor): Quantized representations.
                  Shape: (b, n_tokens_per_quantizer, z_dim).
                - indices (torch.Tensor): Quantization indices.
                  Shape: (b, n_tokens_per_quantizer) for VQ or (b, n_tokens_per_quantizer, n_quantizers) for RVQ.
                - perplexity (Union[float, List[float]]): Codebook perplexity.
                  Float for single quantizer, List[float] for multiple quantizers.
                - commit_loss (torch.Tensor): Commitment loss scalar tensor.
        """
        if isinstance(self.vq, ResidualVectorQuantize):
            z_q, indices, _, commitment_loss, codebook_loss = self.vq(z_e)
            commit_loss = commitment_loss.mean() + codebook_loss.mean()
        elif isinstance(self.vq, VectorQuantize):
            z_q, indices, commit_loss = self.vq(z_e)
        else:
            raise NotImplementedError(f"VQ type {type(self.vq)} not implemented")

        if return_perplexity:
            if len(indices.size()) < 3:
                indices = indices.unsqueeze(-1)
            perplexity = []
            for k in range(indices.size(-1)):
                this_indices = indices[:, :, k]
                indices_count = torch.bincount(
                    this_indices.view(-1), minlength=self.vq.codebook_size
                )
                if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                    torch.distributed.all_reduce(indices_count)
                this_avg_probs = indices_count.float() / indices_count.sum()
                perplexity.append(
                    ((-(this_avg_probs * torch.log(this_avg_probs + 1e-10)).sum()).exp().item())
                )
        else:
            perplexity = 0

        return z_q, indices, perplexity, commit_loss

    def _dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """Dequantize token indices back to continuous latent representations.

        Args:
            indices (torch.Tensor): Quantization indices. Shape depends on quantizer type:
                - For VQ: (b, n_tokens) or (b, n_tokens, 1)
                - For RVQ: (b, n_tokens_per_quantizer, n_quantizers)

        Returns:
            torch.Tensor: Dequantized latent representations.
                Shape: (b, n_tokens_per_quantizer, z_dim)
        """
        if self.num_quantizers == 1:
            if len(indices.size()) == 3:
                indices = indices.squeeze(-1)
        if isinstance(self.vq, ResidualVectorQuantize):
            z_q = self.vq.from_codes(indices)[0]
        elif isinstance(self.vq, VectorQuantize):
            z_q = self.vq.get_output_from_indices(indices)
        else:
            raise NotImplementedError(f"VQ type {type(self.vq)} not implemented in _dequantize")
        return z_q

    def _decode(
        self, z_q: torch.Tensor, embodiment_ids: torch.Tensor = None, durations: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode quantized latent representations into action sequences.

        Args:
            z_q (torch.Tensor): Quantized latent representations.
                Shape: (b, n_tokens_per_quantizer, z_dim).
            embodiment_ids (Union[torch.Tensor, int, None], optional): Embodiment IDs.
                Shape: (b,) if tensor. If int, the same embodiment ID is used for all
                sequences. Defaults to None, which uses `self.default_embodiment_id`.
            durations (torch.Tensor | None, optional): Duration of each action sequence in seconds.
                Shape: (b,). If None, uses default duration from embodiment_config.
                Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - x_recon (torch.Tensor): Reconstructed action sequences.
                  Shape: (b, seq_len, max_action_dim).
                - padding_mask (torch.Tensor): Padding mask indicating valid timesteps.
                  Shape: (b, seq_len), where True indicates valid timesteps.
        """
        embodiment_ids = (
            embodiment_ids if embodiment_ids is not None else self.default_embodiment_id
        )
        x_recon, padding_mask = self.decoder(z_q, embodiment_ids, durations)
        return x_recon, padding_mask

    @torch.no_grad()
    def encode(
        self,
        x: Union[np.ndarray, torch.Tensor],
        embodiment_ids: Union[List[int], int, None] = None,
        padding_mask: Union[List[bool], np.ndarray, torch.Tensor, None] = None,
        **kwargs,
    ) -> List[List[int]]:
        """Encode action sequences into latent representations (token indices).

        This method converts action sequences into discrete token indices using the encoder
        and vector quantizer. The input can be either a numpy array or torch tensor.

        Args:
            x (Union[np.ndarray, torch.Tensor]): Action sequences to encode.
                Shape: (b, seq_len, max_action_dim).
                Assumes that the action dimension is zero-padded to the max action dimension.
                `seq_len` is supposed to be `int(duration * freq)` for each embodiment and
                padded to the max sequence length.
            embodiment_ids (Union[List[int], int, None], optional): Embodiment IDs.
                Shape: (b,) if list. If int, the same embodiment ID is repeated for all
                sequences in the batch. It specifies the embodiment to encode.
                Defaults to None, which uses `self.default_embodiment_id`.
            padding_mask (Union[List[bool], np.ndarray, torch.Tensor, None], optional):
                Padding mask, where `False` values indicate padding. Shape: (b, seq_len).
                Defaults to None. It is used to mask the padding tokens on `seq_len` dimension.
            **kwargs: Additional keyword arguments (currently unused, reserved for future use).

        Returns:
            List[List[int]]: List of token sequences. Shape: (b, n_tokens), where n_tokens
            is determined by the model configuration (typically `config.n_tokens`).

        Raises:
            ValueError: If input shapes are invalid or incompatible with the model configuration.
            TypeError: If input types are not supported.

        Examples:
            >>> import numpy as np
            >>> # Using numpy array
            >>> x = np.random.randn(2, 10, 7).astype(np.float32)
            >>> tokens = model.encode(x, embodiment_ids=[0, 0])
            >>> # Using torch tensor
            >>> x_tensor = torch.randn(2, 10, 7)
            >>> tokens = model.encode(x_tensor, embodiment_ids=[0, 0])
        """
        self.eval()

        # Validate and convert input x
        if isinstance(x, np.ndarray):
            if x.ndim != 3:
                raise ValueError(
                    f"Expected 3D input array (batch, seq_len, action_dim), got {x.ndim}D array with shape {x.shape}"
                )
            x_tensor = torch.tensor(x, dtype=self.dtype, device=self.device)
        elif isinstance(x, torch.Tensor):
            if x.ndim != 3:
                raise ValueError(
                    f"Expected 3D tensor (batch, seq_len, action_dim), got {x.ndim}D tensor with shape {x.shape}"
                )
            x_tensor = x.to(dtype=self.dtype, device=self.device)
        else:
            raise TypeError(f"Input x must be numpy.ndarray or torch.Tensor, got {type(x)}")

        # Validate batch size
        batch_size = x_tensor.shape[0]
        if batch_size == 0:
            raise ValueError("Batch size must be at least 1")

        # Handle embodiment_ids
        embodiment_ids = (
            embodiment_ids if embodiment_ids is not None else self.default_embodiment_id
        )
        if isinstance(embodiment_ids, int):
            if not 0 <= embodiment_ids < len(self.config.embodiment_config):
                raise ValueError(
                    f"embodiment_id {embodiment_ids} is out of range [0, {len(self.config.embodiment_config)}). "
                    f"Available embodiment IDs: {list(range(len(self.config.embodiment_config)))}"
                )
            embodiment_ids_tensor = torch.tensor(
                [embodiment_ids] * batch_size, dtype=torch.long, device=self.device
            )
        elif isinstance(embodiment_ids, list):
            if len(embodiment_ids) != batch_size:
                raise ValueError(
                    f"Length of embodiment_ids ({len(embodiment_ids)}) must match batch size ({batch_size})"
                )
            for eid in embodiment_ids:
                if not isinstance(eid, int) or not 0 <= eid < len(self.config.embodiment_config):
                    raise ValueError(
                        f"Invalid embodiment_id {eid}. Must be an integer in range [0, {len(self.config.embodiment_config)})"
                    )
            embodiment_ids_tensor = torch.tensor(
                embodiment_ids, dtype=torch.long, device=self.device
            )
        else:
            raise TypeError(
                f"embodiment_ids must be int, List[int], or None, got {type(embodiment_ids)}"
            )

        # Handle padding_mask
        padding_mask_tensor = None
        if padding_mask is not None:
            if isinstance(padding_mask, (list, np.ndarray)):
                padding_mask_tensor = torch.tensor(
                    padding_mask, dtype=torch.bool, device=self.device
                )
            elif isinstance(padding_mask, torch.Tensor):
                padding_mask_tensor = padding_mask.to(dtype=torch.bool, device=self.device)
            else:
                raise TypeError(
                    f"padding_mask must be List[bool], np.ndarray, torch.Tensor, or None, got {type(padding_mask)}"
                )
            if padding_mask_tensor.shape != (batch_size, x_tensor.shape[1]):
                raise ValueError(
                    f"padding_mask shape {padding_mask_tensor.shape} does not match expected shape "
                    f"({batch_size}, {x_tensor.shape[1]})"
                )

        with torch.no_grad():
            z_e = self._encode(x_tensor, embodiment_ids_tensor, padding_mask_tensor)
            _, indices, _, _ = self._quantize(z_e, return_perplexity=False)

            # Reshape indices: for RVQ, indices shape is (b, n, s), for VQ it's (b, n)
            if len(indices.size()) > 2:
                codes_list = einops.rearrange(indices, "b n s -> b (s n)").cpu()
            else:
                codes_list = indices.cpu()

            codes_list = codes_list.tolist()
            return codes_list

    @torch.no_grad()
    def decode(
        self,
        tokens: Union[List[List[int]], np.ndarray, torch.Tensor],
        embodiment_ids: Union[List[int], int, None] = None,
        durations: Union[List[float], np.ndarray, torch.Tensor, None] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Decode token sequences into action sequences.

        This method reconstructs action sequences from discrete token indices using the
        vector quantizer and decoder. The input tokens can be a list of lists, numpy array,
        or torch tensor.

        Args:
            tokens (Union[List[List[int]], np.ndarray, torch.Tensor]): Token sequences to decode.
                Shape: (b, n_tokens), where n_tokens must be divisible by `n_tokens_per_quantizer`.
                For RVQ, tokens are interleaved: [q0_t0, q1_t0, ..., qN_t0, q0_t1, ...].
            embodiment_ids (Union[List[int], int, None], optional): Embodiment IDs.
                Shape: (b,) if list. If int, the same embodiment ID is repeated for all
                sequences in the batch. It specifies the embodiment to decode.
                Defaults to None, which uses `self.default_embodiment_id`.
            durations (Union[List[float], np.ndarray, torch.Tensor, None], optional):
                Duration of each action sequence in seconds. Shape: (b,).
                If None, the duration is inferred from the default values in `embodiment_config`.
                Defaults to None.
            **kwargs: Additional keyword arguments (currently unused, reserved for future use).

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - reconstructed_actions: Reconstructed action sequences.
                  Shape: (b, seq_len, max_action_dim).
                - padding_mask: Padding mask indicating valid timesteps.
                  Shape: (b, seq_len), where True indicates valid timesteps.

        Raises:
            ValueError: If token sequence length is invalid or incompatible with the model configuration.
            TypeError: If input types are not supported.

        Examples:
            >>> # Using list of lists
            >>> tokens = [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]]
            >>> actions, mask = model.decode(tokens, embodiment_ids=[0, 0])
            >>> # Using numpy array
            >>> tokens_np = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
            >>> actions, mask = model.decode(tokens_np, embodiment_ids=[0, 0])
            >>> # Using torch tensor
            >>> tokens_tensor = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
            >>> actions, mask = model.decode(tokens_tensor, embodiment_ids=[0, 0])
        """
        self.eval()

        # Validate and convert input tokens
        if isinstance(tokens, list):
            if not all(isinstance(seq, list) for seq in tokens):
                raise TypeError("If tokens is a list, all elements must be lists")
            if len(tokens) == 0:
                raise ValueError("Tokens list cannot be empty")
            if not all(isinstance(val, (int, np.integer)) for seq in tokens for val in seq):
                raise TypeError("All token values must be integers")
            tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=self.device)
        elif isinstance(tokens, np.ndarray):
            if tokens.ndim != 2:
                raise ValueError(
                    f"Expected 2D array (batch, n_tokens), got {tokens.ndim}D array with shape {tokens.shape}"
                )
            if not np.issubdtype(tokens.dtype, np.integer):
                raise TypeError(f"Tokens array must have integer dtype, got {tokens.dtype}")
            tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=self.device)
        elif isinstance(tokens, torch.Tensor):
            if tokens.ndim != 2:
                raise ValueError(
                    f"Expected 2D tensor (batch, n_tokens), got {tokens.ndim}D tensor with shape {tokens.shape}"
                )
            if not tokens.dtype.is_integer:
                raise TypeError(f"Tokens tensor must have integer dtype, got {tokens.dtype}")
            tokens_tensor = tokens.to(dtype=torch.long, device=self.device)
        else:
            raise TypeError(
                f"tokens must be List[List[int]], np.ndarray, or torch.Tensor, got {type(tokens)}"
            )

        batch_size, n_tokens = tokens_tensor.shape
        if batch_size == 0:
            raise ValueError("Batch size must be at least 1")
        if n_tokens == 0:
            raise ValueError("Token sequence length must be at least 1")

        # Validate token sequence length
        if n_tokens % self.n_tokens_per_quantizer != 0:
            raise ValueError(
                f"Token sequence length ({n_tokens}) must be divisible by tokens per quantizer "
                f"({self.n_tokens_per_quantizer}). Total tokens: {n_tokens}, "
                f"Expected multiple of: {self.n_tokens_per_quantizer}. "
                f"Number of quantizers: {self.num_quantizers}, Total tokens per sequence: {self.config.n_tokens}"
            )

        # Validate token values are within codebook range
        if tokens_tensor.min() < 0 or tokens_tensor.max() >= self.vocab_size:
            raise ValueError(
                f"Token values must be in range [0, {self.vocab_size}), "
                f"got range [{tokens_tensor.min().item()}, {tokens_tensor.max().item()}]"
            )

        # Handle embodiment_ids
        embodiment_ids = (
            embodiment_ids if embodiment_ids is not None else self.default_embodiment_id
        )
        if isinstance(embodiment_ids, int):
            if not 0 <= embodiment_ids < len(self.config.embodiment_config):
                raise ValueError(
                    f"embodiment_id {embodiment_ids} is out of range [0, {len(self.config.embodiment_config)}). "
                    f"Available embodiment IDs: {list(range(len(self.config.embodiment_config)))}"
                )
            embodiment_ids_tensor = torch.tensor(
                [embodiment_ids] * batch_size, dtype=torch.long, device=self.device
            )
        elif isinstance(embodiment_ids, list):
            if len(embodiment_ids) != batch_size:
                raise ValueError(
                    f"Length of embodiment_ids ({len(embodiment_ids)}) must match batch size ({batch_size})"
                )
            for eid in embodiment_ids:
                if not isinstance(eid, int) or not 0 <= eid < len(self.config.embodiment_config):
                    raise ValueError(
                        f"Invalid embodiment_id {eid}. Must be an integer in range [0, {len(self.config.embodiment_config)})"
                    )
            embodiment_ids_tensor = torch.tensor(
                embodiment_ids, dtype=torch.long, device=self.device
            )
        else:
            raise TypeError(
                f"embodiment_ids must be int, List[int], or None, got {type(embodiment_ids)}"
            )

        # Handle durations
        durations_tensor = None
        if durations is not None:
            if isinstance(durations, (list, np.ndarray)):
                durations_tensor = torch.tensor(durations, dtype=torch.float32, device=self.device)
            elif isinstance(durations, torch.Tensor):
                durations_tensor = durations.to(dtype=torch.float32, device=self.device)
            else:
                raise TypeError(
                    f"durations must be List[float], np.ndarray, torch.Tensor, or None, got {type(durations)}"
                )
            if durations_tensor.ndim != 1:
                raise ValueError(
                    f"durations must be 1D, got {durations_tensor.ndim}D with shape {durations_tensor.shape}"
                )
            if len(durations_tensor) != batch_size:
                raise ValueError(
                    f"Length of durations ({len(durations_tensor)}) must match batch size ({batch_size})"
                )
            if (durations_tensor <= 0).any():
                raise ValueError("All durations must be positive")

        # Reshape tokens for dequantization: (b, n_tokens) -> (b, n_tokens_per_quantizer, n_quantizers)
        indices = einops.rearrange(tokens_tensor, "b (n m) -> b m n", m=self.n_tokens_per_quantizer)

        with torch.no_grad():
            z_q = self._dequantize(indices)
            x_recon, padding_mask = self._decode(z_q, embodiment_ids_tensor, durations_tensor)

        return x_recon.float().cpu().numpy(), padding_mask.float().cpu().numpy()

    def forward(
        self,
        x: Union[torch.Tensor, np.ndarray],
        embodiment_ids: Union[torch.Tensor, int, List[int], None] = None,
        padding_mask: Union[torch.Tensor, List[bool], np.ndarray, None] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the full ActionCodec pipeline.

        This method performs encoding, quantization, and decoding in a single forward pass.
        It is primarily used during training to compute reconstruction loss and commitment loss.
        Both numpy arrays and torch tensors are supported as input.

        Args:
            x (Union[torch.Tensor, np.ndarray]): Action sequences to process.
                Shape: (b, seq_len, max_action_dim).
            embodiment_ids (Union[torch.Tensor, int, List[int], None], optional):
                Embodiment IDs. Shape: (b,) if tensor or list. If int, same ID for all sequences.
                Defaults to None, which uses `self.default_embodiment_id`.
            padding_mask (Union[torch.Tensor, List[bool], np.ndarray, None], optional):
                Padding mask. Shape: (b, seq_len). Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - x_recon (torch.Tensor): Reconstructed action sequences.
                  Shape: (b, seq_len, max_action_dim).
                - recon_mask (torch.Tensor): Reconstruction mask indicating valid timesteps.
                  Shape: (b, seq_len), where True indicates valid timesteps.

        Note:
            - For inference use cases, prefer using `encode()` and `decode()` methods separately.
            - If you need token indices, use the `encode()` method instead.
        """
        # Convert numpy array to torch tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=self.dtype, device=self.device)

        # Handle embodiment_ids conversion
        if isinstance(embodiment_ids, list):
            embodiment_ids = torch.tensor(embodiment_ids, device=x.device, dtype=torch.long)
        elif isinstance(embodiment_ids, int):
            # Keep as int, will be handled by _encode
            pass

        # Handle padding_mask conversion
        if isinstance(padding_mask, (list, np.ndarray)):
            padding_mask = torch.tensor(padding_mask, device=x.device, dtype=torch.bool)

        # Full forward pass: encode -> quantize -> decode
        z_e = self._encode(x, embodiment_ids, padding_mask)
        z_q, _, _, _ = self._quantize(z_e, return_perplexity=False)
        x_recon, recon_mask = self._decode(z_q, embodiment_ids)

        return x_recon, recon_mask


AutoModel.register(ActionCodecConfig, ActionCodec)

__all__ = ["ActionCodec"]


if __name__ == "__main__":
    print("=== ActionCodec Comprehensive Test ===\n")

    # 1. Configuration Setup (RVQ enabled with n_quantizers=4)
    initial_config = {
        "robot_A": {"action_dim": 7, "freq": 10, "duration": 1, "description": "Robot A"},
    }

    # We set n_quantizers=4 to test Residual VQ logic
    config = ActionCodecConfig(
        embodiment_config=initial_config,
        n_tokens=16,  # Total tokens per sequence (latent_len * n_quantizers)
        n_quantizers=4,  # RVQ depth
        vq_type="rvq",
        vq_codebook_size=256,
        encoder_dim=128,
        decoder_dim=128,
    )

    # Expected latent sequence length = n_tokens / n_quantizers = 16 / 4 = 4
    latent_seq_len = int(config.n_tokens // config.n_quantizers)
    print(
        f"Config: {config.n_quantizers} quantizers, {latent_seq_len} latent vectors per sequence."
    )

    codec = ActionCodec(config)
    codec.eval()

    # 2. Basic Encode/Decode Test
    print("\n--- Test 1: Basic Encode/Decode ---")
    batch_size = 2
    seq_len_A = 10  # 10Hz * 1s

    # Create random action data for Robot A (ID 0)
    x = np.random.randn(batch_size, seq_len_A, 7).astype(np.float32)
    # Masking: Second item in batch is half padding
    padding_mask = np.ones((batch_size, seq_len_A), dtype=bool)
    padding_mask[1, 5:] = False

    embodiment_ids = [0, 0]

    # Encode
    codes = codec.encode(x, embodiment_ids, padding_mask)
    print(f"Encoded codes shape (list length): {len(codes)} x {len(codes[0])}")

    # Validate code length
    assert len(codes[0]) == config.n_tokens, (
        f"Expected {config.n_tokens} tokens, got {len(codes[0])}"
    )

    # Decode
    x_recon, recon_mask = codec.decode(codes, embodiment_ids)
    print(f"Reconstructed shape: {x_recon.shape}")
    print(f"Recon mask shape: {recon_mask.shape}")

    assert x_recon.shape == (batch_size, seq_len_A, 7)  # Should imply zero-padding to max dim 7

    # 3. Expansion Test
    print("\n--- Test 2: Dynamic Expansion ---")
    new_robot_config = {
        "robot_B": {"action_dim": 10, "freq": 20, "duration": 1, "description": "Robot B (Larger)"}
    }

    print("Expanding codec to include Robot B (10 dims, 20Hz)...")
    codec.expand_embodiment(new_robot_config)

    assert codec.encoder.max_action_dim == 10
    assert codec.decoder.max_action_dim == 10
    print("✅ Expansion successful.")

    # 4. Mixed Batch Test (Old + New Robot)
    print("\n--- Test 3: Mixed Batch Inference ---")

    # Batch: [Robot A, Robot B]
    # Robot A: 10Hz, 1s -> 10 steps. Dims 7.
    # Robot B: 20Hz, 1s -> 20 steps. Dims 10.
    # Batch Max Steps: 20. Batch Max Dims: 10.

    batch_x_mixed = np.zeros((2, 20, 10), dtype=np.float32)

    # Fill Robot A data (index 0)
    data_A = np.random.randn(10, 7)
    batch_x_mixed[0, :10, :7] = data_A

    # Fill Robot B data (index 1)
    data_B = np.random.randn(20, 10)
    batch_x_mixed[1, :20, :10] = data_B

    # Embodiment IDs: 0 for A, 1 for B
    # Note: expand_embodiment appends. Original was 0, new is 1.
    mixed_ids = [0, 1]

    # Encode Mask
    mixed_mask = np.zeros((2, 20), dtype=bool)
    mixed_mask[0, :10] = True
    mixed_mask[1, :20] = True

    print("Encoding mixed batch...")
    mixed_codes = codec.encode(batch_x_mixed, mixed_ids, mixed_mask)

    print("Decoding mixed batch...")
    # Explicit durations (optional, but good for verification if we wanted to override defaults)
    durations = [1, 1]
    x_recon_mixed, dec_mask_mixed = codec.decode(mixed_codes, mixed_ids, durations)

    print(f"Mixed Recon Shape: {x_recon_mixed.shape}")

    # Validation
    # Robot A output check (mask should be True for first 10, False for rest)
    valid_A = dec_mask_mixed[0].sum()
    valid_B = dec_mask_mixed[1].sum()

    print(f"Valid steps detected by Decoder: Robot A={valid_A}, Robot B={valid_B}")

    assert valid_A == 10
    assert valid_B == 20

    # Check dimensionality preservation
    # Robot A's reconstruction in dims 7-9 should be noise or zero (depending on implementation),
    # but dims 0-6 should contain signal.
    print("✅ Mixed batch processed successfully.")

    print("\n✨ All systems go.")
