from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp


SCRIPTS = Path(__file__).resolve().parents[1] / "src" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from predict_magworld_vcc2026_v3 import bayesian_decode, calibrate_effect


def test_calibration_preserves_full_self_effect() -> None:
    delta = np.asarray([0.2, -0.8, 0.4, -0.1], dtype=np.float32)

    calibrated = calibrate_effect(
        delta,
        target_index=1,
        top_k=1,
        downstream_scale=0.25,
        self_scale=1.0,
        max_delta=1.5,
    )

    assert calibrated[1] == delta[1]
    assert np.count_nonzero(calibrated) == 2
    assert calibrated[2] == 0.25 * delta[2]


def test_bayesian_decode_is_sparse_and_directional() -> None:
    base = sp.csr_matrix(
        np.asarray(
            [
                [0, 5, 0, 2],
                [1, 8, 4, 2],
                [0, 3, 2, 2],
            ],
            dtype=np.uint32,
        )
    )
    delta = np.asarray([3.0, -0.7, 0.0, 0.0], dtype=np.float32)
    gene_mean = np.asarray(base.mean(0)).ravel()

    decoded = bayesian_decode(
        base,
        delta,
        gene_mean,
        np.random.default_rng(17),
        prior_strength=2.0,
    ).toarray()
    original = base.toarray()

    assert np.all(decoded[:, 1] <= original[:, 1])
    assert decoded[:, 0].sum() > original[:, 0].sum()
    assert np.array_equal(decoded[:, 2:], original[:, 2:])


def test_zero_effect_leaves_counts_unchanged() -> None:
    base = sp.csr_matrix(np.asarray([[0, 2], [3, 1]], dtype=np.uint32))
    decoded = bayesian_decode(
        base,
        np.zeros(2, dtype=np.float32),
        np.asarray(base.mean(0)).ravel(),
        np.random.default_rng(3),
        prior_strength=2.0,
    )

    assert (decoded != base).nnz == 0
