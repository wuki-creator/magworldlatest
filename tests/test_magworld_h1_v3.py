from __future__ import annotations

import sys
from pathlib import Path

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "src" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from model_world_h1_v3 import ReciprocalMagneticField, WorldModelH1V3


def test_single_gene_charge_has_gradient() -> None:
    torch.manual_seed(7)
    field = ReciprocalMagneticField(d_model=8, d_z=4)
    embeddings = torch.nn.functional.normalize(torch.randn(11, 8), dim=-1)

    force, distance_sq, charge = field(embeddings, torch.tensor([3]))
    loss = force.square().sum()
    loss.backward()

    assert force.shape == (1, 11)
    assert distance_sq.shape == (1, 11)
    assert charge.shape == (1,)
    assert field.source_charge.weight.grad is not None
    assert float(field.source_charge.weight.grad.abs().sum()) > 0


def test_pairwise_magnetic_force_is_reciprocal() -> None:
    torch.manual_seed(9)
    field = ReciprocalMagneticField(d_model=8, d_z=4)
    embeddings = torch.nn.functional.normalize(torch.randn(11, 8), dim=-1)

    force, _, _ = field(embeddings, torch.arange(len(embeddings)))

    assert torch.allclose(force, force.T, atol=1e-6)


def test_world_model_is_control_anchored_and_target_conditioned() -> None:
    torch.manual_seed(11)
    model = WorldModelH1V3(
        n_genes=13,
        d_model=8,
        d_z=4,
        d_hidden=12,
    )
    model.initialize_gene_embeddings(torch.randn(13, 8))
    control = torch.rand(2, 13)
    targets = torch.tensor([2, 9])

    delta = model.predict_delta(control, targets)
    prediction = model(control, targets)

    assert delta.shape == control.shape
    assert torch.allclose(prediction, control + delta)
    assert not torch.allclose(delta[0], delta[1])
    assert torch.all(delta.abs() <= model.max_delta + 1e-6)
