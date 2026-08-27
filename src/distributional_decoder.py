"""Distribution-preserving count calibration and blind single-cell diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.stats import wasserstein_distance


def heterogeneous_bayesian_decode(
    base: sp.csr_matrix,
    delta: np.ndarray,
    gene_mean: np.ndarray,
    count_rng: np.random.Generator,
    response_rng: np.random.Generator,
    prior_strength: float,
    response_sigma: float,
) -> sp.csr_matrix:
    """Decode a correlated per-cell perturbation-strength distribution."""
    if response_sigma <= 0:
        raise ValueError("response sigma must be positive")
    if prior_strength <= 0:
        raise ValueError("prior strength must be positive")
    selected = np.flatnonzero(np.abs(delta) > 1e-8)
    if len(selected) == 0:
        return base.copy().astype(np.uint32)

    original = base[:, selected].toarray().astype(np.int64, copy=False)
    decoded = original.copy()
    strength = np.exp(
        response_rng.normal(
            loc=-0.5 * response_sigma**2,
            scale=response_sigma,
            size=(base.shape[0], 1),
        )
    )
    strength /= max(float(strength.mean()), 1e-8)
    fold_change = np.exp(strength * delta[selected][None, :])
    down = delta[selected] < 0
    if np.any(down):
        decoded[:, down] = count_rng.binomial(
            original[:, down], np.clip(fold_change[:, down], 0.0, 1.0)
        )
    up = delta[selected] > 0
    if np.any(up):
        prior_mean = np.maximum(gene_mean[selected[up]], 1e-4)
        posterior_shape = prior_strength * prior_mean[None, :] + original[:, up]
        posterior_rate_sample = count_rng.gamma(
            shape=posterior_shape, scale=1.0 / (prior_strength + 1.0)
        )
        added = count_rng.poisson(
            posterior_rate_sample * np.maximum(fold_change[:, up] - 1.0, 0.0)
        )
        decoded[:, up] = original[:, up] + added

    base_coo = base.tocoo()
    selected_mask = np.zeros(base.shape[1], dtype=bool)
    selected_mask[selected] = True
    keep = ~selected_mask[base_coo.col]
    changed_rows, changed_columns = np.nonzero(decoded)
    changed_values = decoded[changed_rows, changed_columns]
    values = np.concatenate(
        [base_coo.data[keep].astype(np.uint32), changed_values.astype(np.uint32)]
    )
    rows = np.concatenate([base_coo.row[keep], changed_rows])
    columns = np.concatenate([base_coo.col[keep], selected[changed_columns]])
    return sp.csr_matrix(
        (values, (rows, columns)), shape=base.shape, dtype=np.uint32
    )


def match_library_sizes(
    decoded: sp.csr_matrix,
    reference: sp.csr_matrix,
    strength: float,
    rng: np.random.Generator,
) -> sp.csr_matrix:
    """Move decoded library sizes toward matched controls without changing support."""
    if decoded.shape != reference.shape:
        raise ValueError("decoded and reference matrices must have the same shape")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("library match strength must be between zero and one")
    matrix = decoded.tocsr().astype(np.uint32)
    if strength == 0.0:
        return matrix.copy()

    decoded_totals = np.asarray(matrix.sum(1)).ravel().astype(np.int64)
    reference_totals = np.asarray(reference.sum(1)).ravel().astype(np.int64)
    targets = np.rint(
        (1.0 - strength) * decoded_totals + strength * reference_totals
    ).astype(np.int64)
    data: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    indptr = [0]
    for row, target_total in enumerate(targets):
        start, stop = matrix.indptr[row : row + 2]
        row_indices = matrix.indices[start:stop]
        counts = matrix.data[start:stop].astype(np.int64, copy=True)
        current_total = int(counts.sum())
        target_total = int(max(target_total, 0))
        if current_total and target_total < current_total:
            counts = rng.multivariate_hypergeometric(counts, target_total)
        elif current_total and target_total > current_total:
            counts += rng.multinomial(
                target_total - current_total, counts / current_total
            )
        keep = counts > 0
        data.append(counts[keep].astype(np.uint32))
        indices.append(row_indices[keep])
        indptr.append(indptr[-1] + int(np.sum(keep)))
    return sp.csr_matrix(
        (
            np.concatenate(data) if data else np.empty(0, dtype=np.uint32),
            np.concatenate(indices) if indices else np.empty(0, dtype=np.int32),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=matrix.shape,
        dtype=np.uint32,
    )


def _log_cp10k(matrix: sp.csr_matrix, columns: np.ndarray) -> np.ndarray:
    values = matrix[:, columns].toarray().astype(np.float64)
    library = np.asarray(matrix.sum(1)).ravel().astype(np.float64).clip(min=1.0)
    return np.log1p(values * (10_000.0 / library)[:, None])


def _top_indices(values: np.ndarray, count: int) -> np.ndarray:
    count = min(max(count, 1), len(values))
    return np.argpartition(values, -count)[-count:]


@dataclass(frozen=True)
class DistributionReference:
    features: np.ndarray
    center: np.ndarray
    components: np.ndarray
    component_scale: np.ndarray
    truth_projection: np.ndarray
    truth_library: np.ndarray
    truth_detected: np.ndarray


def build_distribution_reference(
    base: sp.csr_matrix,
    truth: sp.csr_matrix,
    predicted_effect: np.ndarray,
    max_features: int = 256,
    n_components: int = 32,
) -> DistributionReference:
    """Build a truth-independent feature basis and cache the truth projection."""
    if base.shape[1] != len(predicted_effect) or truth.shape[1] != len(predicted_effect):
        raise ValueError("effect and matrix gene dimensions differ")
    mean = np.asarray(base.mean(0)).ravel()
    second = np.asarray(base.power(2).mean(0)).ravel()
    variance = np.maximum(second - np.square(mean), 0.0)
    half = max(1, max_features // 2)
    features = np.unique(
        np.concatenate(
            [
                _top_indices(variance, half),
                _top_indices(np.abs(predicted_effect), half),
            ]
        )
    )
    base_log = _log_cp10k(base, features)
    center = base_log.mean(0)
    centered = base_log - center
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    component_count = min(n_components, len(singular_values), len(features))
    components = right[:component_count]
    component_scale = singular_values[:component_count] / np.sqrt(
        max(len(base_log) - 1, 1)
    )
    component_scale = np.maximum(component_scale, 1e-4)
    truth_projection = (
        (_log_cp10k(truth, features) - center) @ components.T
    ) / component_scale
    return DistributionReference(
        features=features,
        center=center,
        components=components,
        component_scale=component_scale,
        truth_projection=truth_projection,
        truth_library=np.log1p(np.asarray(truth.sum(1)).ravel().astype(np.float64)),
        truth_detected=np.asarray(truth.getnnz(axis=1), dtype=np.float64),
    )


def _gaussian_fid(left: np.ndarray, right: np.ndarray) -> float:
    left_mean = left.mean(0)
    right_mean = right.mean(0)
    left_cov = np.atleast_2d(np.cov(left, rowvar=False))
    right_cov = np.atleast_2d(np.cov(right, rowvar=False))
    eigenvalues, eigenvectors = np.linalg.eigh(left_cov)
    left_sqrt = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))) @ eigenvectors.T
    middle = left_sqrt @ right_cov @ left_sqrt
    middle_eigenvalues = np.linalg.eigvalsh((middle + middle.T) * 0.5)
    covariance_distance = np.trace(left_cov) + np.trace(right_cov) - 2.0 * np.sum(
        np.sqrt(np.maximum(middle_eigenvalues, 0.0))
    )
    return float(max(np.sum(np.square(left_mean - right_mean)) + covariance_distance, 0.0))


def distribution_metrics(
    decoded: sp.csr_matrix, reference: DistributionReference
) -> dict[str, float]:
    projected = (
        (_log_cp10k(decoded, reference.features) - reference.center)
        @ reference.components.T
    ) / reference.component_scale
    fid = _gaussian_fid(projected, reference.truth_projection)
    dimension = max(projected.shape[1], 1)
    library = np.log1p(np.asarray(decoded.sum(1)).ravel().astype(np.float64))
    detected = np.asarray(decoded.getnnz(axis=1), dtype=np.float64)
    library_distance = wasserstein_distance(library, reference.truth_library) / max(
        float(np.std(reference.truth_library)), 0.1
    )
    detected_distance = wasserstein_distance(detected, reference.truth_detected) / max(
        float(np.std(reference.truth_detected)), 1.0
    )
    return {
        "latent_fid": fid,
        "latent_fid_similarity": float(np.exp(-fid / dimension)),
        "library_distance": float(library_distance),
        "library_similarity": float(np.exp(-library_distance)),
        "detected_distance": float(detected_distance),
        "detected_similarity": float(np.exp(-detected_distance)),
    }
