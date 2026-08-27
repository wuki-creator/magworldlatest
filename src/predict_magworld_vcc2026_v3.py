"""Generate a sparse Bayesian VCC package candidate from reciprocal MagWorld."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--ensemble-weights", nargs="+", type=float)
    parser.add_argument("--controls-dir", required=True)
    parser.add_argument("--genes", required=True)
    parser.add_argument("--perts", required=True)
    parser.add_argument("--magworld-src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--downstream-scale", type=float, default=0.25)
    parser.add_argument("--self-scale", type=float, default=1.0)
    parser.add_argument("--max-delta", type=float, default=1.5)
    parser.add_argument("--prior-strength", type=float, default=2.0)
    parser.add_argument("--cells-per-target", type=int, default=400)
    parser.add_argument("--prediction-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def read_genes(path: str) -> list[str]:
    lines = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if lines and lines[0].lower() in {"gene_name", "gene"}:
        lines = lines[1:]
    return lines


def adata_genes(adata) -> list[str]:
    if "gene_name" in adata.var:
        return adata.var["gene_name"].astype(str).tolist()
    return adata.var_names.astype(str).tolist()


def aligned_sparse(
    matrix, source_genes: list[str], target_genes: list[str]
) -> sp.csr_matrix:
    values = matrix.tocsr() if sp.issparse(matrix) else sp.csr_matrix(matrix)
    target_lookup = {gene: index for index, gene in enumerate(target_genes)}
    source_columns: list[int] = []
    target_columns: list[int] = []
    for source_index, gene in enumerate(source_genes):
        target_index = target_lookup.get(gene)
        if target_index is not None:
            source_columns.append(source_index)
            target_columns.append(target_index)
    selected = values[:, source_columns].tocoo()
    remapped_columns = np.asarray(target_columns, dtype=np.int32)[selected.col]
    return sp.csr_matrix(
        (selected.data.astype(np.uint32), (selected.row, remapped_columns)),
        shape=(values.shape[0], len(target_genes)),
    )


def choose_rows(adata, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if "ntc_id" not in adata.obs:
        return rng.choice(adata.n_obs, n, replace=adata.n_obs < n)
    groups = adata.obs.groupby("ntc_id", observed=True).indices
    per_group, remainder = divmod(n, len(groups))
    selected: list[int] = []
    for position, key in enumerate(sorted(groups, key=str)):
        rows = np.asarray(groups[key])
        take = per_group + int(position < remainder)
        selected.extend(rng.choice(rows, take, replace=len(rows) < take).tolist())
    return np.asarray(selected, dtype=np.int64)


def log_cp10k_mean(counts: sp.csr_matrix) -> np.ndarray:
    library = np.asarray(counts.sum(1)).ravel()
    normalized = counts.multiply((10_000.0 / np.maximum(library, 1.0))[:, None])
    normalized.data = np.log1p(normalized.data)
    return np.asarray(normalized.mean(0)).ravel().astype(np.float32)


def calibrate_effect(
    delta: np.ndarray,
    target_index: int,
    top_k: int,
    downstream_scale: float,
    self_scale: float,
    max_delta: float,
) -> np.ndarray:
    result = np.zeros_like(delta)
    magnitude = np.abs(delta).copy()
    magnitude[target_index] = -np.inf
    if top_k < 0 or top_k >= len(delta) - 1:
        selected = np.arange(len(delta))
        selected = selected[selected != target_index]
    else:
        selected = np.argpartition(magnitude, -top_k)[-top_k:]
    result[selected] = downstream_scale * delta[selected]
    result[target_index] = self_scale * delta[target_index]
    return np.clip(result, -max_delta, max_delta)


def bayesian_decode(
    base: sp.csr_matrix,
    delta: np.ndarray,
    gene_mean: np.ndarray,
    rng: np.random.Generator,
    prior_strength: float,
) -> sp.csr_matrix:
    """Thin repressed counts and Gamma-Poisson sample induced counts."""
    selected = np.flatnonzero(np.abs(delta) > 1e-8)
    if len(selected) == 0:
        return base.copy().astype(np.uint32)

    original = base[:, selected].toarray().astype(np.int64, copy=False)
    decoded = original.copy()
    fold_change = np.exp(delta[selected])
    down = fold_change < 1.0
    if np.any(down):
        decoded[:, down] = rng.binomial(original[:, down], fold_change[down])
    up = fold_change > 1.0
    if np.any(up):
        prior_mean = np.maximum(gene_mean[selected[up]], 1e-4)
        posterior_shape = prior_strength * prior_mean[None, :] + original[:, up]
        posterior_rate = prior_strength + 1.0
        posterior_rate_sample = rng.gamma(
            shape=posterior_shape,
            scale=1.0 / posterior_rate,
        )
        added = rng.poisson(posterior_rate_sample * (fold_change[up] - 1.0))
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
    return sp.csr_matrix((values, (rows, columns)), shape=base.shape, dtype=np.uint32)


def load_models(
    checkpoint_paths: list[str], model_class, genes: list[str], device: torch.device
) -> list[torch.nn.Module]:
    models: list[torch.nn.Module] = []
    for path in checkpoint_paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint["genes"] != genes:
            raise ValueError(f"checkpoint and official gene list differ: {path}")
        model = model_class(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state"])
        model.to(device).eval()
        models.append(model)
    return models


def predict_effects(
    models: list[torch.nn.Module],
    weights: np.ndarray,
    control_mean: np.ndarray,
    target_indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    prediction = np.zeros((len(target_indices), len(control_mean)), dtype=np.float32)
    control = torch.from_numpy(control_mean).to(device)
    for start in range(0, len(target_indices), batch_size):
        stop = min(start + batch_size, len(target_indices))
        indices = torch.from_numpy(target_indices[start:stop]).to(device)
        controls = control[None, :].expand(stop - start, -1)
        batch_prediction = np.zeros((stop - start, len(control_mean)), dtype=np.float32)
        for weight, model in zip(weights, models, strict=True):
            with torch.no_grad():
                values = model.predict_delta(controls, indices).cpu().numpy()
            batch_prediction += float(weight) * values
        prediction[start:stop] = batch_prediction
    return prediction


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
    model_class = importlib.import_module("model_world_h1_v3").WorldModelH1V3
    models = load_models(args.checkpoints, model_class, genes, device)
    if args.ensemble_weights is None:
        weights = np.full(len(models), 1.0 / len(models), dtype=np.float64)
    else:
        if len(args.ensemble_weights) != len(models):
            raise ValueError("one ensemble weight is required per checkpoint")
        weights = np.asarray(args.ensemble_weights, dtype=np.float64)
        if np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("ensemble weights must be non-negative with a positive sum")
        weights /= weights.sum()

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
            rows = choose_rows(
                controls, args.cells_per_target, args.seed + context_index
            )
            base = aligned_sparse(controls.X[rows], adata_genes(controls), genes)
        finally:
            del controls
        gene_mean = np.asarray(base.mean(0)).ravel().astype(np.float64)
        control_mean = log_cp10k_mean(base)
        raw_effects = predict_effects(
            models,
            weights,
            control_mean,
            target_indices,
            args.prediction_batch_size,
            device,
        )
        delta_l1: list[float] = []
        target_delta: list[float] = []
        library_shift: list[float] = []
        base_library = np.asarray(base.sum(1)).ravel().astype(np.float64)
        for position, (target, target_index) in enumerate(zip(targets, target_indices)):
            delta = calibrate_effect(
                raw_effects[position],
                int(target_index),
                args.top_k,
                args.downstream_scale,
                args.self_scale,
                args.max_delta,
            )
            rng = np.random.default_rng(
                args.seed + 10_000 * context_index + position
            )
            decoded = bayesian_decode(
                base, delta, gene_mean, rng, args.prior_strength
            )
            matrices.append(decoded)
            obs_targets.extend([target] * args.cells_per_target)
            obs_contexts.extend([context] * args.cells_per_target)
            delta_l1.append(float(np.mean(np.abs(delta))))
            target_delta.append(float(delta[target_index]))
            new_library = np.asarray(decoded.sum(1)).ravel().astype(np.float64)
            library_shift.append(
                float(np.median((new_library - base_library) / np.maximum(base_library, 1)))
            )
            if (position + 1) % 25 == 0:
                print(f"context={context} generated={position + 1}/{len(targets)}", flush=True)
        diagnostics[context] = {
            "effect_l1_mean": float(np.mean(delta_l1)),
            "selected_genes": int(args.top_k + 1 if args.top_k >= 0 else len(genes)),
            "target_delta_median": float(np.median(target_delta)),
            "target_delta_positive_fraction": float(np.mean(np.asarray(target_delta) > 0)),
            "library_shift_median": float(np.median(library_shift)),
            "library_shift_p95_abs": float(np.quantile(np.abs(library_shift), 0.95)),
        }

    matrix = sp.vstack(matrices, format="csr", dtype=np.uint32)
    obs = pd.DataFrame(
        {"target_gene": obs_targets, "context": obs_contexts},
        index=[f"magworld_v3_{i:06d}" for i in range(matrix.shape[0])],
    )
    prediction = ad.AnnData(
        X=matrix,
        obs=obs,
        var=pd.DataFrame(index=pd.Index(genes, name="gene_name")),
    )
    prediction.uns["model"] = "MagWorldH1-v3-reciprocal-bayesian"
    prediction.uns["checkpoints"] = [Path(path).name for path in args.checkpoints]
    prediction.uns["ensemble_weights"] = weights.tolist()
    prediction.uns["top_k"] = args.top_k
    prediction.uns["downstream_scale"] = args.downstream_scale
    prediction.uns["self_scale"] = args.self_scale
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
