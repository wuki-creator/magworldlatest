from __future__ import annotations

import numpy as np

from predict_magworld_vcc2026_v5 import (
    center_panel_effects,
    consensus_calibrate_effect,
    consensus_statistics,
    normalize_weights,
)
from evaluate_decoder_h1_v5 import proxy_score


def test_normalize_weights() -> None:
    np.testing.assert_allclose(normalize_weights(np.array([1.0, 2.0]), 2), [1 / 3, 2 / 3])


def test_consensus_statistics_shrinks_disagreement() -> None:
    effects = np.array([[1.0, 1.0], [1.0, -1.0], [1.0, -1.0]], dtype=np.float32)
    mean, agreement, shrinkage = consensus_statistics(effects, np.ones(3), 0.5)
    assert mean[0] == 1.0
    assert agreement[0] == 1.0
    assert shrinkage[0] == 1.0
    assert agreement[1] < 0.5
    assert shrinkage[1] < agreement[1]


def test_consensus_calibration_filters_conflicts_and_gates_down_regulation() -> None:
    effects = np.array(
        [
            [-1.0, 0.8, -0.6, -0.5],
            [-1.0, -0.8, -0.7, -0.5],
            [-1.0, 0.7, -0.8, -0.5],
        ],
        dtype=np.float32,
    )
    calibrated, diagnostics = consensus_calibrate_effect(
        effects,
        np.ones(3),
        np.array([2.0, 2.0, 0.0, 2.0]),
        target_index=0,
        top_k=3,
        downstream_scale=1.0,
        self_scale=1.2,
        max_delta=2.0,
        min_sign_agreement=0.5,
        uncertainty_penalty=0.0,
        expression_gate_scale=0.5,
    )
    assert calibrated[0] == -1.2
    assert calibrated[1] == 0.0
    assert calibrated[2] == 0.0
    assert calibrated[3] < 0.0
    assert diagnostics["selected_genes"] == 3.0


def test_proxy_score_penalizes_bad_lfc_accuracy() -> None:
    row = {
        "cosine": 0.2,
        "lfc_nmae": 0.5,
        "direction_fidelity": 0.8,
        "signed_reach": 0.2,
        "reach": 0.25,
        "jaccard": 0.1,
        "predicted_de": 100.0,
        "truth_de": 100.0,
    }
    good = proxy_score(row)
    row["lfc_nmae"] = 1.5
    assert proxy_score(row) < good


def test_panel_centering_removes_shared_response() -> None:
    effects = np.array(
        [[[2.0, 1.0], [2.0, -1.0], [2.0, 0.0]]], dtype=np.float32
    )
    centered = center_panel_effects(effects, 1.0)
    np.testing.assert_allclose(centered.mean(axis=1), 0.0, atol=1e-7)
    np.testing.assert_allclose(center_panel_effects(effects, 0.0), effects)
