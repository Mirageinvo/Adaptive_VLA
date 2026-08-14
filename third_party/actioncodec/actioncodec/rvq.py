from typing import List, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from vector_quantize_pytorch import VectorQuantize as torchVQ


def sample_vectors(samples, num):
    # samples: (N, D), num_samples: N, feature dim: D
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)
    return samples[indices].float()  # (num, D), ensure fp32


def ema_inplace(moving_avg, new, decay):
    # moving_avg: (codebook_size) or (codebook_size, D'), new: same as moving_avg
    """Update exponential moving average in-place"""
    moving_avg.data.mul_(decay).add_(new.float(), alpha=(1 - decay))  # ensure fp32


def kmeans(samples, num_clusters, num_iters=10):
    # samples: (N, D), N samples with D dimensions
    dim, _ = samples.shape[-1], torch.float32  # Force fp32
    means = sample_vectors(samples, num_clusters).float()  # (num_clusters, D), ensure fp32

    for _ in range(num_iters):
        dists = -(
            samples.float().pow(2).sum(1, keepdim=True)  # (N, 1), ensure fp32
            - 2 * samples.float() @ means.t()  # (N, num_clusters), ensure fp32
            + means.t().float().pow(2).sum(0, keepdim=True)
        )  # (1, num_clusters), ensure fp32
        # dists: (N, num_clusters)
        buckets = dists.max(dim=-1).indices  # (N)
        bins = torch.bincount(buckets, minlength=num_clusters)  # (num_clusters)
        zero_mask = bins == 0  # (num_clusters)
        bins_min_clamped = bins.masked_fill(zero_mask, 1)  # (num_clusters)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=torch.float32)  # (num_clusters, D), ensure fp32
        new_means.scatter_add_(
            0, buckets.unsqueeze(1).expand(-1, dim), samples.float()
        )  # (num_clusters, D), ensure fp32
        new_means = new_means / bins_min_clamped[..., None]  # (num_clusters, D)
        means = torch.where(zero_mask[..., None], means, new_means)  # (num_clusters, D)

    # Final cluster assignments for returning cluster sizes
    dists = -(
        samples.float().pow(2).sum(1, keepdim=True)
        - 2 * samples.float() @ means.t()
        + means.t().float().pow(2).sum(0, keepdim=True)
    )  # (N, num_clusters), ensure fp32
    buckets = dists.max(dim=-1).indices  # (N)
    bins = torch.bincount(buckets, minlength=num_clusters).float()  # (num_clusters), ensure fp32

    return means, bins  # (num_clusters, D), (num_clusters)


class VectorQuantize(nn.Module):
    def __init__(
        self,
        input_dim,
        codebook_size,
        codebook_dim,
        commitment=1.0,
        decay=0.99,  # EMA decay
        epsilon=1e-5,  # Laplace smoothing epsilon
        threshold_ema_dead=2,  # Dead code threshold
        kmeans_init=True,  # Use kmeans initialization
        kmeans_iters=10,  # Kmeans iterations
        rotation_trick=False,  # Use rotation trick
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment = commitment
        self.decay = decay
        self.epsilon = epsilon
        self.threshold_ema_dead = threshold_ema_dead
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.rotation_trick = rotation_trick

        if self.input_dim != self.codebook_dim:
            self.in_project = nn.Linear(input_dim, codebook_dim)
            self.out_project = nn.Linear(codebook_dim, input_dim)
        else:
            self.in_project = nn.Identity()
            self.out_project = nn.Identity()

        # Initialize codebook and EMA buffers
        init_fn = torch.zeros if kmeans_init else lambda x, y: torch.randn(x, y)
        self.register_buffer(
            "codebook", init_fn(codebook_size, codebook_dim).float()
        )  # (codebook_size, D'), ensure fp32
        self.register_buffer("inited", torch.tensor([not kmeans_init], dtype=torch.bool))  # (1)
        self.register_buffer("cluster_size", torch.zeros(codebook_size).float())  # (codebook_size), ensure fp32
        self.register_buffer("embed_avg", self.codebook.clone().float())  # (codebook_size, D'), ensure fp32

    def ema_update(self, encodings, embed_onehot):
        # encodings: (B*T, D'), embed_onehot: (B*T, codebook_size)
        """Update codebook using EMA"""
        encodings = encodings.float()  # Ensure fp32
        embed_onehot = embed_onehot.float()  # Ensure fp32
        cluster_size_new = embed_onehot.sum(0)  # (codebook_size)
        embed_sum = encodings.t() @ embed_onehot  # (D', codebook_size)

        # Distributed reduction
        if dist.is_initialized():
            dist.all_reduce(cluster_size_new, op=dist.ReduceOp.SUM)
            dist.all_reduce(embed_sum, op=dist.ReduceOp.SUM)

        ema_inplace(self.cluster_size, cluster_size_new, self.decay)  # (codebook_size)
        ema_inplace(self.embed_avg, embed_sum.t(), self.decay)  # (codebook_size, D')

        # Laplace smoothing
        cluster_size = (self.cluster_size + self.epsilon) / (
            self.cluster_size.sum() + self.codebook_size * self.epsilon
        )  # (codebook_size)
        cluster_size = cluster_size * self.cluster_size.sum()  # (codebook_size)
        self.codebook.copy_(self.embed_avg / cluster_size.unsqueeze(1))  # (codebook_size, D')

    def replace_dead_codes(self, encodings):
        # encodings: (B*T, D')
        """Replace dead codes with random samples from current batch"""
        if self.threshold_ema_dead == 0:
            return

        dead_mask = self.cluster_size < self.threshold_ema_dead  # (codebook_size)
        if dead_mask.any():
            if dist.is_initialized() and dist.get_rank() == 0:
                samples = sample_vectors(encodings.float(), self.codebook_size)  # (codebook_size, D'), ensure fp32
                print(f"Replace {dead_mask.sum().item()} dead codes")
            else:
                samples = torch.zeros_like(self.codebook).float()  # Placeholder, ensure fp32

            # Broadcast samples
            if dist.is_initialized():
                dist.broadcast(samples, src=0)

            self.codebook[dead_mask] = samples[: dead_mask.sum()].to(self.codebook.dtype)  # Update dead codes

    def init_codebook(self, encodings):
        # encodings: (B*T, D')
        """Initialize codebook with k-means and update cluster_size"""
        if self.inited.item():
            return

        if dist.is_initialized() and dist.get_rank() == 0:
            embed, cluster_sizes = kmeans(
                encodings.float(), self.codebook_size, self.kmeans_iters
            )  # (codebook_size, D'), (codebook_size), ensure fp32
        else:
            embed = torch.zeros(self.codebook_size, self.codebook_dim, device=encodings.device).float()  # ensure fp32
            cluster_sizes = torch.zeros(self.codebook_size, device=encodings.device, dtype=torch.float32)  # ensure fp32

        # Broadcast results
        if dist.is_initialized():
            dist.broadcast(embed, src=0)
            dist.broadcast(cluster_sizes, src=0)

        self.codebook.copy_(embed)  # (codebook_size, D')
        self.embed_avg.copy_(embed.clone())  # (codebook_size, D')
        self.cluster_size.copy_(cluster_sizes.float())  # (codebook_size)
        self.inited.fill_(True)

    def forward(self, z):
        self = self.to(torch.float32)
        z = z.float()
        z_e = self.in_project(z).float()

        # Rearrange for quantization
        encodings = rearrange(z_e, "b t d -> (b t) d").float()  # (B*T, D'), ensure fp32

        # Initialize codebook if needed
        if self.kmeans_init and not self.inited.item():
            self.init_codebook(encodings)

        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ self.codebook.float().t()
            + self.codebook.float().pow(2).sum(1, keepdim=True).t()
        )
        indices = (-dist).max(1)[1]

        # cosine_similarity = F.cosine_similarity(encodings[None], self.codebook[:, None], dim=-1)
        # indices = cosine_similarity.max(dim=0)[1]

        indices = rearrange(indices, "(b t) -> b t", b=z.size(0))
        z_q = self.decode_code(indices).float()
        commit_loss = F.mse_loss(z_e, z_q.detach()) * self.commitment

        if self.training and torch.is_grad_enabled():
            embed_onehot = F.one_hot(indices.view(-1), self.codebook_size).float()
            self.ema_update(encodings, embed_onehot)
            self.replace_dead_codes(encodings)

        z_q = (z_q - z_e).detach() + z_e
        z_q = self.out_project(z_q).float()

        return (
            z_q,
            commit_loss,
            torch.tensor(0.0, device=z.device, dtype=torch.float32),
            indices,
            z_e,
        )

    def decode_code(self, embed_id):  # embed_id: (B, T)
        return F.embedding(embed_id, self.codebook).float()  # (B, D', T), ensure fp32


# class VectorQuantize(nn.Module):
#     """
#     Implementation of VQ similar to Karpathy's repo:
#     https://github.com/karpathy/deep-vector-quantization
#     Additionally uses following tricks from Improved VQGAN
#     (https://arxiv.org/pdf/2110.04627.pdf):
#         1. Factorized codes: Perform nearest neighbor lookup in low-dimensional space
#             for improved codebook usage
#         2. l2-normalized codes: Converts euclidean distance to cosine similarity which
#             improves training stability
#     """

#     def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
#         super().__init__()
#         self.codebook_size = codebook_size
#         self.codebook_dim = codebook_dim

#         self.in_proj = nn.Linear(input_dim, codebook_dim)
#         self.out_proj = nn.Linear(codebook_dim, input_dim)
#         self.codebook = nn.Embedding(codebook_size, codebook_dim)

#     def forward(self, z: torch.Tensor):
#         """
#         Args:
#             z (torch.Tensor): shape (b, t, d)

#         Returns:
#             z_q (torch.Tensor): shape (b, t, d)
#             commitment_loss (torch.Tensor): shape (1)
#             codebook_loss (torch.Tensor): shape (1)
#             indices (torch.Tensor): shape (b, t)
#             z_e (torch.Tensor): shape (b, t, d)
#         """

#         # Factorized codes (ViT-VQGAN) Project input into low-dimensional space
#         z_e = self.in_proj(z)
#         z_q, indices = self.decode_latents(z_e)

#         commitment_loss = F.mse_loss(z_e, z_q.detach()) * 0.25
#         codebook_loss = F.mse_loss(z_q, z_e.detach())

#         z_q = z_e + (z_q - z_e).detach()  # noop in forward pass, straight-through gradient estimator in backward pass

#         z_q = self.out_proj(z_q)

#         return z_q, commitment_loss, codebook_loss, indices, z_e

#     def embed_code(self, embed_id):
#         return F.embedding(embed_id, self.codebook.weight)

#     def decode_code(self, embed_id):
#         return self.embed_code(embed_id)

#     def decode_latents(self, latents: torch.Tensor):
#         codebook = self.codebook.weight
#         encodings = rearrange(latents, "b t d -> (b t) d")

#         cosine_similarity = F.cosine_similarity(encodings[None], codebook[:, None], dim=-1)
#         indices = cosine_similarity.max(dim=0)[1]
#         indices = rearrange(indices, "(b t) -> b t", b=latents.size(0))

#         # encodings = F.normalize(encodings)
#         # codebook = F.normalize(codebook)
#         # dist = (
#         #     encodings.pow(2).sum(1, keepdim=True)
#         #     - 2 * encodings @ codebook.t()
#         #     + codebook.pow(2).sum(1, keepdim=True).t()
#         # )
#         # indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))

#         z_q = self.decode_code(indices)
#         return z_q, indices


class ResidualVectorQuantize(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        n_codebooks: int = 4,
        codebook_size: int = 512,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.25,
        commitment: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_ema_dead: int = 2,
        kmeans_init: bool = True,
        kmeans_iters: int = 10,
        rotation_trick: bool = False,
    ):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]

        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        self.quantizers = nn.ModuleList(
            [
                VectorQuantize(
                    input_dim=dim,
                    codebook_size=codebook_size,
                    codebook_dim=codebook_dim[i],
                    commitment=commitment,
                    decay=decay,
                    epsilon=epsilon,
                    threshold_ema_dead=threshold_ema_dead,
                    kmeans_init=kmeans_init,
                    kmeans_iters=kmeans_iters,
                    rotation_trick=rotation_trick,
                )
                for i in range(n_codebooks)
            ]
        )
        self.quantizer_dropout = quantizer_dropout

    def forward(self, z, n_quantizers: int = None):
        """Quantized the input tensor using a fixed set of `n` codebooks and returns
        the corresponding codebook vectors
        Parameters
        ----------
        z : Tensor[B x D x T]
        n_quantizers : int, optional
            No. of quantizers to use
            (n_quantizers < self.n_codebooks ex: for quantizer dropout)
            Note: if `self.quantizer_dropout` is True, this argument is ignored
                when in training mode, and a random number of quantizers is used.
        Returns
        -------
        dict
            A dictionary with the following keys:

            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
        """
        z_q, residual = 0, z
        commitment_loss, codebook_loss = 0, 0

        codebook_indices, latents = [], []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        if self.training:
            n_quantizers = torch.ones((z.shape[0],)) * self.n_codebooks + 1
            dropout = torch.randint(1, self.n_codebooks + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)

        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break

            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(residual)

            # Create mask to apply quantizer dropout
            mask = torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i

            # Sum losses
            commitment_loss += (commitment_loss_i * mask).mean()
            codebook_loss += (codebook_loss_i * mask).mean()

            codebook_indices.append(indices_i)
            latents.append(z_e_i)

        codes = torch.stack(codebook_indices, dim=-1)
        latents = torch.cat(latents, dim=1)

        return z_q, codes, latents, commitment_loss, codebook_loss

    def from_codes(self, codes: torch.Tensor):
        """Given the quantized codes, reconstruct the continuous representation
        Parameters
        ----------
        codes : Tensor[B x N x T]
            Quantized discrete representation of input
        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        """
        z_q = 0.0
        z_p = []
        n_codebooks = codes.shape[-1]
        for i in range(n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[..., i])
            z_p.append(z_p_i)

            z_q_i = self.quantizers[i].out_project(z_p_i)
            z_q = z_q + z_q_i
        return z_q, torch.cat(z_p, dim=-1), codes

    def from_latents(self, latents: torch.Tensor):
        """Given the unquantized latents, reconstruct the
        continuous representation after quantization.

        Parameters
        ----------
        latents : Tensor[B x N x T]
            Continuous representation of input after projection

        Returns
        -------
        Tensor[B x D x T]
            Quantized representation of full-projected space
        Tensor[B x D x T]
            Quantized representation of latent space
        """
        z_q = 0
        z_p = []
        codes = []
        dims = np.cumsum([0] + [q.codebook_dim for q in self.quantizers])

        n_codebooks = np.where(dims <= latents.shape[1])[0].max(axis=0, keepdims=True)[0]
        for i in range(n_codebooks):
            j, k = dims[i], dims[i + 1]
            z_p_i, codes_i = self.quantizers[i].decode_latents(latents[:, j:k, :])
            z_p.append(z_p_i)
            codes.append(codes_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i

        return z_q, torch.cat(z_p, dim=1), torch.stack(codes, dim=1)


class IndependentVectorQuantize(nn.Module):
    def __init__(self, num_codebooks: int = 1, **kwargs):
        super().__init__()
        self.vector_quantizers = nn.ModuleList([torchVQ(**kwargs) for _ in range(num_codebooks)])
        self.num_codebooks = num_codebooks
        self.codebook_size = self.vector_quantizers[0].codebook_size

    @property
    def ema_update(self):
        return [vq.ema_update for vq in self.vector_quantizers]

    @property
    def codebook(self):
        return torch.stack([vq.codebook for vq in self.vector_quantizers], dim=0)

    @codebook.setter
    def codebook(self, codes: List[torch.Tensor]):
        assert len(codes) == self.num_codebooks, "Number of codebooks must match"
        if not self.separate_codebook_per_head:
            codes = rearrange(codes, "... -> 1 ...")

        for i, code in enumerate(codes):
            self.vector_quantizers[i].codebook.copy_(code)

    def get_codes_from_indices(self, indices: torch.Tensor):
        codes = list()
        for i in range(self.num_codebooks):
            codes.append(self.vector_quantizers[i].get_codes_from_indices(indices[..., i : i + 1]))
        return torch.cat(codes, dim=-2)

    def get_output_from_indices(self, indices: torch.Tensor):
        outputs = list()
        for i in range(self.num_codebooks):
            outputs.append(self.vector_quantizers[i].get_output_from_indices(indices[..., i : i + 1]))
        return torch.cat(outputs, dim=-2)

    def update_in_place_optimizer(self):
        for i in range(self.num_codebooks):
            self.vector_quantizers[i].update_in_place_optimizer()

    def forward(self, x: torch.Tensor, *args, **kwargs):
        assert x.shape[1] == self.num_codebooks
        quantized, indices, commit_losses = list(), list(), 0
        for i in range(self.num_codebooks):
            quantized_i, indices_i, commit_loss_i = self.vector_quantizers[i](x[:, i : i + 1])
            quantized.append(quantized_i)
            indices.append(indices_i)
            commit_losses += commit_loss_i
        quantized = torch.cat(quantized, dim=-2)
        indices = torch.cat(indices, dim=-1)
        return quantized, indices, commit_losses / self.num_codebooks


if __name__ == "__main__":
    vq = IndependentVectorQuantize(
        num_codebooks=16,
        dim=256,
        codebook_size=2048,
        decay=0.8,  # the exponential moving average decay, lower means the dictionary will change faster
        commitment_weight=1.0,  # the weight on the commitment loss
    )

    x = torch.randn(1, 16, 256)
    quantized, indices, commit_loss = vq(x)  # (1, 1024, 256), (1, 1024), (1)
