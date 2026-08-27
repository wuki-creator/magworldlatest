from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from distributional_decoder import (
    build_distribution_reference,
    distribution_metrics,
    heterogeneous_bayesian_decode,
    match_library_sizes,
)
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


def test_library_matching_is_optional_and_matches_reference_totals() -> None:
    decoded = sp.csr_matrix([[4, 2, 0], [0, 3, 3]], dtype=np.uint32)
    reference = sp.csr_matrix([[2, 1, 0], [0, 5, 7]], dtype=np.uint32)
    unchanged = match_library_sizes(
        decoded, reference, 0.0, np.random.default_rng(7)
    )
    np.testing.assert_array_equal(unchanged.toarray(), decoded.toarray())
    matched = match_library_sizes(
        decoded, reference, 1.0, np.random.default_rng(7)
    )
    np.testing.assert_array_equal(
        np.asarray(matched.sum(1)).ravel(), np.asarray(reference.sum(1)).ravel()
    )


def test_distribution_metric_prefers_identical_cells() -> None:
    base = sp.csr_matrix(
        [[5, 1, 0, 2], [3, 2, 1, 1], [4, 0, 2, 2], [2, 3, 1, 2]],
        dtype=np.uint32,
    )
    truth = sp.csr_matrix(
        [[3, 3, 0, 2], [2, 3, 1, 1], [3, 1, 2, 2], [1, 4, 1, 2]],
        dtype=np.uint32,
    )
    reference = build_distribution_reference(
        base, truth, np.array([0.0, 1.0, 0.0, 0.0]), max_features=4, n_components=3
    )
    identical = distribution_metrics(truth, reference)
    shifted = distribution_metrics(base, reference)
    assert identical["latent_fid"] <= 1e-8
    assert identical["latent_fid_similarity"] > shifted["latent_fid_similarity"]


def test_heterogeneous_decoder_preserves_shape_and_integer_counts() -> None:
    base = sp.csr_matrix(
        [[5, 1, 0], [3, 0, 2], [4, 2, 1]], dtype=np.uint32
    )
    decoded = heterogeneous_bayesian_decode(
        base,
        np.array([-0.5, 0.4, 0.0]),
        np.asarray(base.mean(0)).ravel(),
        np.random.default_rng(11),
        np.random.default_rng(13),
        prior_strength=2.0,
        response_sigma=0.3,
    )
    assert decoded.shape == base.shape
    assert decoded.dtype == np.uint32
    assert np.all(decoded.data >= 0)
