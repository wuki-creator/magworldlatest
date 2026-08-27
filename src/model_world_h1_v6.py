"""Sparse-context directed magnetic world model for H1/VCC experiments."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_world_h1_v4 import WorldModelH1V4


class ParametricChebyshevSparseEncoder(nn.Module):
    """Sparse top-k encoder with a learnable Chebyshev-style value filter.

    The recurrence acts on signed, normalized token amplitudes. It provides a
    bounded multi-order polynomial filter without densifying the single-cell
    profile or requiring an unverified external gene graph.
    """

    def __init__(self, n_genes: int, d_z: int, d_hidden: int, top_k: int,
                 cheb_order: int = 3) -> None:
        super().__init__()
        if top_k <= 0 or top_k > n_genes:
            raise ValueError("sparse top_k must be between one and n_genes")
        self.n_genes = n_genes
        self.top_k = top_k
        if cheb_order < 1:
            raise ValueError("cheb_order must be positive")
        self.cheb_order = cheb_order
        self.gene_tokens = nn.Embedding(n_genes, d_z)
        self.value_projection = nn.Sequential(
            nn.Linear(1, d_z), nn.SiLU(), nn.Linear(d_z, d_z)
        )
        self.output = nn.Sequential(
            nn.Linear(d_z + 3, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_z),
            nn.LayerNorm(d_z),
        )
        self.cheb_coeff = nn.Parameter(torch.zeros(cheb_order + 1))
        nn.init.constant_(self.cheb_coeff[0], 1.0)

    def forward(self, x_ctrl: torch.Tensor) -> torch.Tensor:
        values, indices = torch.topk(
            x_ctrl.abs(), k=self.top_k, dim=1, largest=True, sorted=False
        )
        signed_values = torch.gather(x_ctrl, 1, indices)
        tokens = self.gene_tokens(indices)
        amplitudes = signed_values / values.max(dim=1, keepdim=True).values.clamp_min(1e-6)
        basis_prev = tokens
        filtered = self.cheb_coeff[0] * basis_prev
        if self.cheb_order >= 1:
            basis = amplitudes.unsqueeze(-1) * tokens
            filtered = filtered + self.cheb_coeff[1] * basis
            for order in range(2, self.cheb_order + 1):
                basis_next = 2.0 * amplitudes.unsqueeze(-1) * basis - basis_prev
                filtered = filtered + self.cheb_coeff[order] * basis_next
                basis_prev, basis = basis, basis_next
        tokens = filtered * self.value_projection(signed_values.unsqueeze(-1))
        weights = values.sum(1, keepdim=True).clamp_min(1e-6)
        pooled = tokens.sum(1) / weights
        mean = x_ctrl.mean(1, keepdim=True)
        std = x_ctrl.std(1, keepdim=True, unbiased=False)
        nonzero = (x_ctrl != 0).float().mean(1, keepdim=True)
        return self.output(torch.cat([pooled, mean, std, nonzero], dim=1))


class SparseContextEncoder(ParametricChebyshevSparseEncoder):
    """Backward-compatible name for the sparse Chebyshev encoder."""


class WorldModelH1V6(WorldModelH1V4):
    """V4 magnetic field with sparse context and field-level context modulation."""

    def __init__(self, *args, sparse_top_k: int = 2048, cheb_order: int = 3, **kwargs) -> None:
        d_hidden = kwargs.get("d_hidden", args[3] if len(args) > 3 else 192)
        super().__init__(*args, **kwargs)
        self.sparse_top_k = sparse_top_k
        self.context_encoder = SparseContextEncoder(
            self.n_genes, self.d_z, d_hidden,
            sparse_top_k,
            cheb_order,
        )

    def predict_delta(
        self,
        x_ctrl: torch.Tensor,
        pert_idx: torch.Tensor | list[torch.Tensor],
    ) -> torch.Tensor:
        indices = self._indices(pert_idx).to(x_ctrl.device)
        embeddings = self.gene_emb.weight
        magnetic_force, _, _ = self.magnet(embeddings, indices)
        source = self.source_code(embeddings[indices])
        receiver = self.receiver_code(embeddings) + self.receiver_residual
        interaction = source @ receiver.T / math.sqrt(self.d_z)
        context_input = (
            F.layer_norm(x_ctrl, (self.n_genes,))
            if self.normalize_context
            else x_ctrl
        )
        context = self.context_encoder(context_input)
        context_receiver = F.normalize(self.context_receiver(embeddings), dim=-1)
        context_gate = self.context_strength * torch.tanh(
            context @ context_receiver.T / math.sqrt(self.d_z)
        )
        magnetic_weight = torch.sigmoid(self.magnetic_mix)
        pair_field = magnetic_weight * magnetic_force + (1.0 - magnetic_weight) * interaction
        shared_bias_scale = self.effective_shared_bias_scale()
        # Preserve v4's validated gate semantics with the sparse context code.
        raw_delta = (
            F.softplus(self.gene_scale)[None, :] * pair_field
            + shared_bias_scale * self.gene_bias[None, :]
        ) * (1.0 + context_gate)
        self_effect = -F.softplus(self.raw_self_effect)
        raw_delta = raw_delta.scatter_add(
            1, indices[:, None], self_effect.expand(len(indices), 1)
        )
        return self.max_delta * torch.tanh(raw_delta / self.max_delta)


WorldModelH1V3 = WorldModelH1V6
