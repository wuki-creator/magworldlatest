"""Reciprocal, gene-conditioned magnetic model for zero-shot perturbations."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReciprocalMagneticField(nn.Module):
    """Pair a perturbation magnet with every response-gene magnet."""

    def __init__(self, d_model: int, d_z: int) -> None:
        super().__init__()
        self.source_charge = nn.Linear(d_model, 1)
        self.receiver_charge = nn.Linear(d_model, 1)
        self.source_pole = nn.Linear(d_model, d_z)
        self.receiver_pole = nn.Linear(d_model, d_z)
        self.source_bias = nn.Parameter(torch.tensor(-0.5))
        self.raw_distance_scale = nn.Parameter(torch.tensor(0.0))

    def forward(
        self, gene_emb: torch.Tensor, pert_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source_charge = F.softplus(
            self.source_charge(gene_emb).squeeze(-1) + self.source_bias
        )
        receiver_charge = torch.tanh(self.receiver_charge(gene_emb).squeeze(-1))
        source_pole = F.normalize(self.source_pole(gene_emb), dim=-1)
        receiver_pole = F.normalize(self.receiver_pole(gene_emb), dim=-1)

        forward_alignment = source_pole[pert_idx] @ receiver_pole.T
        reverse_alignment = receiver_pole[pert_idx] @ source_pole.T
        forward_distance_sq = (2.0 - 2.0 * forward_alignment).clamp_min(1e-4)
        reverse_distance_sq = (2.0 - 2.0 * reverse_alignment).clamp_min(1e-4)
        distance_scale = F.softplus(self.raw_distance_scale)
        forward_force = (
            source_charge[pert_idx, None]
            * receiver_charge[None, :]
            * forward_alignment
            / (1.0 + distance_scale * forward_distance_sq)
        )
        reverse_force = (
            receiver_charge[pert_idx, None]
            * source_charge[None, :]
            * reverse_alignment
            / (1.0 + distance_scale * reverse_distance_sq)
        )
        force = 0.5 * (forward_force + reverse_force)
        distance_sq = 0.5 * (forward_distance_sq + reverse_distance_sq)
        return force, distance_sq, source_charge[pert_idx]


class WorldModelH1V3(nn.Module):
    """Transfer perturbation effects through reciprocal gene-level fields."""

    def __init__(
        self,
        n_genes: int,
        d_model: int = 128,
        d_z: int = 64,
        d_hidden: int = 192,
        max_delta: float = 1.5,
        context_strength: float = 0.25,
        freeze_gene_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.d_model = d_model
        self.d_z = d_z
        self.max_delta = max_delta
        self.context_strength = context_strength
        self.gene_emb = nn.Embedding(n_genes, d_model)
        if freeze_gene_embedding:
            self.gene_emb.weight.requires_grad_(False)

        self.magnet = ReciprocalMagneticField(d_model, d_z)
        self.source_code = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_z),
            nn.LayerNorm(d_z),
        )
        self.receiver_code = nn.Sequential(
            nn.Linear(d_model, d_z),
            nn.LayerNorm(d_z),
        )
        self.receiver_residual = nn.Parameter(torch.zeros(n_genes, d_z))
        self.context_encoder = nn.Sequential(
            nn.Linear(n_genes, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_z),
            nn.LayerNorm(d_z),
        )
        self.context_receiver = nn.Linear(d_model, d_z, bias=False)
        self.gene_scale = nn.Parameter(torch.full((n_genes,), -1.5))
        self.gene_bias = nn.Parameter(torch.zeros(n_genes))
        self.magnetic_mix = nn.Parameter(torch.tensor(0.0))
        self.raw_self_effect = nn.Parameter(torch.tensor(-0.4))

    @torch.no_grad()
    def initialize_gene_embeddings(self, features: torch.Tensor) -> None:
        if features.shape != self.gene_emb.weight.shape:
            raise ValueError(
                f"expected gene features {tuple(self.gene_emb.weight.shape)}, "
                f"got {tuple(features.shape)}"
            )
        normalized = F.normalize(features.float(), dim=-1)
        self.gene_emb.weight.copy_(normalized)

    @staticmethod
    def _indices(pert_idx: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        if isinstance(pert_idx, torch.Tensor):
            return pert_idx.reshape(-1).long()
        return torch.stack([value.reshape(-1)[0] for value in pert_idx]).long()

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

        context = self.context_encoder(x_ctrl)
        context_receiver = F.normalize(self.context_receiver(embeddings), dim=-1)
        context_gate = self.context_strength * torch.tanh(
            context @ context_receiver.T / math.sqrt(self.d_z)
        )
        magnetic_weight = torch.sigmoid(self.magnetic_mix)
        pair_field = magnetic_weight * magnetic_force + (1.0 - magnetic_weight) * interaction
        raw_delta = (
            F.softplus(self.gene_scale)[None, :] * pair_field
            + self.gene_bias[None, :]
        ) * (1.0 + context_gate)

        self_effect = -F.softplus(self.raw_self_effect)
        raw_delta = raw_delta.scatter_add(
            1,
            indices[:, None],
            self_effect.expand(len(indices), 1),
        )
        return self.max_delta * torch.tanh(raw_delta / self.max_delta)

    def forward(
        self,
        x_ctrl: torch.Tensor,
        pert_idx: torch.Tensor | list[torch.Tensor],
    ) -> torch.Tensor:
        return x_ctrl + self.predict_delta(x_ctrl, pert_idx)

    @torch.no_grad()
    def magnetic_diagnostics(self, pert_idx: torch.Tensor) -> dict[str, float]:
        force, distance_sq, source_charge = self.magnet(
            self.gene_emb.weight, self._indices(pert_idx)
        )
        return {
            "source_charge_mean": float(source_charge.mean()),
            "force_abs_mean": float(force.abs().mean()),
            "force_abs_max": float(force.abs().max()),
            "distance_mean": float(distance_sq.sqrt().mean()),
            "magnetic_mix": float(torch.sigmoid(self.magnetic_mix)),
            "self_effect": float(-F.softplus(self.raw_self_effect)),
        }
