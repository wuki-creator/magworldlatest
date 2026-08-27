"""Generate robust-consensus VCC predictions from MagWorld ensembles."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from predict_magworld_vcc2026_v4 import (
    adata_genes,
    aligned_sparse,
    bayesian_decode,
    choose_rows,
    load_models,
    log_cp10k_mean,
    read_genes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--ensemble-weights", nargs="+", type=float)
    parser.add_argument("--controls-dir", required=True)
    parser.add_argument("--genes", required=True)
    parser.add_argument("--perts", required=True)
    parser.add_argument("--magworld-src", required=True)
    parser.add_argument("--model-module", default="model_world_h1_v4")
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--downstream-scale", type=float, default=2.0)
    parser.add_argument("--self-scale", type=float, default=1.38)
    parser.add_argument("--max-delta", type=float, default=2.0)
    parser.add_argument("--prior-strength", type=float, default=2.0)
    parser.add_argument("--min-sign-agreement", type=float, default=0.0)
    parser.add_argument("--uncertainty-penalty", type=float, default=0.0)
    parser.add_argument(
        "--panel-centering",
        type=float,
        default=0.0,
        help="Fraction of the 300-target shared response removed before decoding.",
    )
    parser.add_argument(
        "--expression-gate-scale",
        type=float,
        default=0.0,
        help="Raw mean-count scale for down-regulation availability; 0 disables.",
    )
    parser.add_argument("--cells-per-target", type=int, default=400)
    parser.add_argument("--prediction-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def normalize_weights(weights: np.ndarray, model_count: int) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    if len(values) != model_count or np.any(values < 0) or values.sum() <= 0:
        raise ValueError("invalid ensemble weights")
    return values / values.sum()


def predict_model_effects(
    models: list[torch.nn.Module],
    control_mean: np.ndarray,
    target_indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Return each model's effects instead of discarding ensemble uncertainty."""
    prediction = np.zeros(
        (len(models), len(target_indices), len(control_mean)), dtype=np.float32
    )
    control = torch.from_numpy(control_mean).to(device)
    for start in range(0, len(target_indices), batch_size):
        stop = min(start + batch_size, len(target_indices))
        indices = torch.from_numpy(target_indices[start:stop]).to(device)
        controls = control[None, :].expand(stop - start, -1)
        for model_index, model in enumerate(models):
            with torch.no_grad():
                prediction[model_index, start:stop] = (
                    model.predict_delta(controls, indices).cpu().numpy()
                )
    return prediction


def center_panel_effects(
    model_effects: np.ndarray, panel_centering: float
) -> np.ndarray:
    """Remove response components shared by every perturbation in the panel."""
    if model_effects.ndim != 3:
        raise ValueError("model_effects must have shape (models, targets, genes)")
    if not 0.0 <= panel_centering <= 1.0:
        raise ValueError("panel_centering must be between zero and one")
    shared = model_effects.mean(axis=1, keepdims=True)
    return model_effects - panel_centering * shared


def consensus_statistics(
    model_effects: np.ndarray,
    weights: np.ndarray,
    uncertainty_penalty: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean effect, directional agreement, and uncertainty shrinkage."""
    if model_effects.ndim != 2:
        raise ValueError("model_effects must have shape (models, genes)")
    if uncertainty_penalty < 0:
        raise ValueError("uncertainty_penalty must be non-negative")
    normalized = normalize_weights(weights, model_effects.shape[0])
    mean = np.sum(model_effects * normalized[:, None], axis=0)
    sign_agreement = np.abs(
        np.sum(np.sign(model_effects) * normalized[:, None], axis=0)
    )
    variance = np.sum(
        normalized[:, None] * np.square(model_effects - mean[None, :]), axis=0
    )
    relative_std = np.sqrt(np.maximum(variance, 0.0)) / (
        np.abs(mean) + np.float32(1e-4)
    )
    shrinkage = sign_agreement / (1.0 + uncertainty_penalty * relative_std)
    return mean.astype(np.float32), sign_agreement.astype(np.float32), shrinkage.astype(np.float32)


def consensus_calibrate_effect(
    model_effects: np.ndarray,
    weights: np.ndarray,
    gene_mean: np.ndarray,
    target_index: int,
    top_k: int,
    downstream_scale: float,
    self_scale: float,
    max_delta: float,
    min_sign_agreement: float,
    uncertainty_penalty: float,
    expression_gate_scale: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Select stable genes and calibrate effects for the count decoder."""
    if not 0.0 <= min_sign_agreement <= 1.0:
        raise ValueError("min_sign_agreement must be between zero and one")
    if expression_gate_scale < 0:
        raise ValueError("expression_gate_scale must be non-negative")
    mean, sign_agreement, shrinkage = consensus_statistics(
        model_effects, weights, uncertainty_penalty
    )
    stable_effect = mean * shrinkage
    selection_score = np.abs(stable_effect)
    selection_score[sign_agreement < min_sign_agreement] = -np.inf
    selection_score[target_index] = -np.inf
    eligible = np.flatnonzero(np.isfinite(selection_score))
    if top_k < 0 or top_k >= len(eligible):
        selected = eligible
    elif top_k == 0:
        selected = np.empty(0, dtype=np.int64)
    else:
        selected = eligible[
            np.argpartition(selection_score[eligible], -top_k)[-top_k:]
        ]

    result = np.zeros_like(mean)
    result[selected] = downstream_scale * stable_effect[selected]
    if expression_gate_scale > 0:
        negative = selected[result[selected] < 0]
        availability = 1.0 - np.exp(
            -np.maximum(gene_mean[negative], 0.0) / expression_gate_scale
        )
        result[negative] *= availability.astype(result.dtype)

    # CRISPRi target repression is structurally known and should not be shrunk by
    # seed disagreement in downstream response genes.
    result[target_index] = self_scale * mean[target_index]
    result = np.clip(result, -max_delta, max_delta)
    diagnostics = {
        "selected_genes": float(len(selected) + 1),
        "stable_gene_fraction": float(
            np.mean(sign_agreement >= min_sign_agreement)
        ),
        "sign_agreement_mean": float(np.mean(sign_agreement[selected]))
        if len(selected)
        else 0.0,
        "shrinkage_mean": float(np.mean(shrinkage[selected]))
        if len(selected)
        else 0.0,
    }
    return result, diagnostics


def main() -> None:
    args = parse_args()
    started = time.time()
    if args.prior_strength <= 0:
        raise ValueError("prior strength must be positive")
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    genes = read_genes(args.genes)
    module_path = str(Path(args.magworld_src).resolve())
    if module_path not in sys.path:
        sys.path.insert(0, module_path)
    model_class = importlib.import_module(args.model_module).WorldModelH1V3
    models = load_models(args.checkpoints, model_class, genes, device)
    weights = normalize_weights(
        np.ones(len(models)) if args.ensemble_weights is None else args.ensemble_weights,
        len(models),
    )

    gene_lookup = {gene: index for index, gene in enumerate(genes)}
    with open(args.perts, newline="", encoding="utf-8") as handle:
        targets = [row["target_gene"] for row in csv.DictReader(handle)]
    missing = sorted(set(targets) - set(gene_lookup))
    if missing:
        raise ValueError(f"targets missing from official genes: {missing[:10]}")
    target_indices = np.asarray([gene_lookup[target] for target in targets], dtype=np.int64)

    matrices: list[sp.csr_matrix] = []
    obs_targets: list[str] = []
    obs_contexts: list[str] = []
    diagnostics: dict[str, object] = {}
    for context_index, context in enumerate(("A", "B", "C")):
        controls = ad.read_h5ad(Path(args.controls_dir) / f"context_{context}.h5ad")
        try:
            rows = choose_rows(controls, args.cells_per_target, args.seed + context_index)
            base = aligned_sparse(controls.X[rows], adata_genes(controls), genes)
        finally:
            del controls
        gene_mean = np.asarray(base.mean(0)).ravel().astype(np.float64)
        control_mean = log_cp10k_mean(base)
        model_effects = predict_model_effects(
            models,
            control_mean,
            target_indices,
            args.prediction_batch_size,
            device,
        )
        model_effects = center_panel_effects(model_effects, args.panel_centering)
        delta_l1: list[float] = []
        target_delta: list[float] = []
        library_shift: list[float] = []
        sign_agreement: list[float] = []
        shrinkage: list[float] = []
        base_library = np.asarray(base.sum(1)).ravel().astype(np.float64)
        for position, (target, target_index) in enumerate(zip(targets, target_indices)):
            delta, consensus_diag = consensus_calibrate_effect(
                model_effects[:, position],
                weights,
                gene_mean,
                int(target_index),
                args.top_k,
                args.downstream_scale,
                args.self_scale,
                args.max_delta,
                args.min_sign_agreement,
                args.uncertainty_penalty,
                args.expression_gate_scale,
            )
            rng = np.random.default_rng(args.seed + 10_000 * context_index + position)
            decoded = bayesian_decode(base, delta, gene_mean, rng, args.prior_strength)
            matrices.append(decoded)
            obs_targets.extend([target] * args.cells_per_target)
            obs_contexts.extend([context] * args.cells_per_target)
            delta_l1.append(float(np.mean(np.abs(delta))))
            target_delta.append(float(delta[target_index]))
            sign_agreement.append(consensus_diag["sign_agreement_mean"])
            shrinkage.append(consensus_diag["shrinkage_mean"])
            new_library = np.asarray(decoded.sum(1)).ravel().astype(np.float64)
            library_shift.append(
                float(np.median((new_library - base_library) / np.maximum(base_library, 1)))
            )
            if (position + 1) % 25 == 0:
                print(f"context={context} generated={position + 1}/{len(targets)}", flush=True)
        diagnostics[context] = {
            "effect_l1_mean": float(np.mean(delta_l1)),
            "selected_genes": int(args.top_k + 1),
            "target_delta_median": float(np.median(target_delta)),
            "target_delta_positive_fraction": float(np.mean(np.asarray(target_delta) > 0)),
            "sign_agreement_mean": float(np.mean(sign_agreement)),
            "shrinkage_mean": float(np.mean(shrinkage)),
            "library_shift_median": float(np.median(library_shift)),
            "library_shift_p95_abs": float(np.quantile(np.abs(library_shift), 0.95)),
        }

    matrix = sp.vstack(matrices, format="csr", dtype=np.uint32)
    obs = pd.DataFrame(
        {"target_gene": obs_targets, "context": obs_contexts},
        index=[f"magworld_v5_{i:06d}" for i in range(matrix.shape[0])],
    )
    prediction = ad.AnnData(
        X=matrix,
        obs=obs,
        var=pd.DataFrame(index=pd.Index(genes, name="gene_name")),
    )
    prediction.uns["model"] = "MagWorldH1-v5-robust-consensus"
    prediction.uns["checkpoints"] = [Path(path).name for path in args.checkpoints]
    prediction.uns["ensemble_weights"] = weights.tolist()
    prediction.uns["top_k"] = args.top_k
    prediction.uns["downstream_scale"] = args.downstream_scale
    prediction.uns["self_scale"] = args.self_scale
    prediction.uns["min_sign_agreement"] = args.min_sign_agreement
    prediction.uns["uncertainty_penalty"] = args.uncertainty_penalty
    prediction.uns["panel_centering"] = args.panel_centering
    prediction.uns["expression_gate_scale"] = args.expression_gate_scale
    prediction.uns["diagnostics"] = diagnostics
    prediction.write_h5ad(args.out, compression="gzip", compression_opts=4)
    print(
        json.dumps(
            {
                "output": args.out,
                "shape": list(matrix.shape),
                "nnz": int(matrix.nnz),
                "size": Path(args.out).stat().st_size,
                "seconds": round(time.time() - started, 2),
                "diagnostics": diagnostics,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
