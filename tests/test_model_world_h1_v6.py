from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_world_h1_v6 import SparseContextEncoder, WorldModelH1V6


def test_sparse_context_encoder_is_shape_stable() -> None:
    encoder = SparseContextEncoder(n_genes=12, d_z=4, d_hidden=8, top_k=3)
    values = torch.zeros(2, 12)
    values[0, [1, 5, 9]] = torch.tensor([2.0, -1.0, 0.5])
    values[1, [0, 2]] = torch.tensor([1.0, 3.0])
    encoded = encoder(values)
    assert encoded.shape == (2, 4)
    assert torch.isfinite(encoded).all()


def test_v6_model_produces_finite_target_conditioned_delta() -> None:
    torch.manual_seed(23)
    model = WorldModelH1V6(
        n_genes=13, d_model=8, d_z=4, d_hidden=12, sparse_top_k=4
    )
    model.initialize_gene_embeddings(torch.randn(13, 8))
    control = torch.rand(2, 13)
    delta = model.predict_delta(control, torch.tensor([2, 9]))
    assert delta.shape == control.shape
    assert torch.isfinite(delta).all()
    assert not torch.allclose(delta[0], delta[1])
