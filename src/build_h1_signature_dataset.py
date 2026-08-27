"""Build batch-matched, guide-balanced H1 perturbation signatures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


CONTROL_LABEL = "non-targeting"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h1", nargs="+", required=True)
    parser.add_argument("--genes", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-cells", type=int, default=24)
    parser.add_argument("--min-guide-cells", type=int, default=12)
    return parser.parse_args()


def read_genes(path: str) -> list[str]:
    frame = pd.read_csv(path)
    if frame.shape[1] != 1:
        raise ValueError(f"expected one gene column in {path}")
    return frame.iloc[:, 0].astype(str).tolist()


def gene_names(adata: ad.AnnData) -> np.ndarray:
    if "gene_name" in adata.var:
        return adata.var["gene_name"].astype(str).to_numpy()
    return adata.var_names.astype(str).to_numpy()


def log_cp10k(matrix) -> sp.csr_matrix:
    values = sp.csr_matrix(matrix, dtype=np.float32)
    library = np.asarray(values.sum(1)).ravel().clip(min=1.0)
    values = values.multiply((10_000.0 / library)[:, None]).tocsr()
    values.data = np.log1p(values.data)
    return values


def aligned_mean(
    matrix: sp.csr_matrix,
    rows: np.ndarray,
    source_columns: np.ndarray,
    target_columns: np.ndarray,
    n_genes: int,
) -> np.ndarray:
    result = np.zeros(n_genes, dtype=np.float32)
    if not len(rows):
        return result
    observed = np.asarray(matrix[rows][:, source_columns].mean(0)).ravel()
    result[target_columns] = observed.astype(np.float32, copy=False)
    return result


def matched_control(
    rows: np.ndarray,
    batches: np.ndarray,
    control_by_batch: dict[str, np.ndarray],
    fallback: np.ndarray,
) -> np.ndarray:
    values, counts = np.unique(batches[rows], return_counts=True)
    total = np.zeros_like(fallback)
    used = 0
    for value, count in zip(values, counts):
        control = control_by_batch.get(str(value))
        if control is None:
            continue
        total += int(count) * control
        used += int(count)
    return total / used if used else fallback.copy()


def main() -> None:
    args = parse_args()
    genes = read_genes(args.genes)
    gene_lookup = {gene: index for index, gene in enumerate(genes)}
    n_genes = len(genes)

    target_names: list[str] = []
    target_controls: list[np.ndarray] = []
    target_effects: list[np.ndarray] = []
    target_cells: list[int] = []
    target_guides: list[int] = []
    target_sources: list[str] = []
    replicate_controls: list[np.ndarray] = []
    replicate_effects: list[np.ndarray] = []
    replicate_targets: list[int] = []
    replicate_cells: list[int] = []

    for h1_path in args.h1:
        path = Path(h1_path)
        adata = ad.read_h5ad(path)
        try:
            if not sp.issparse(adata.X):
                raise ValueError(f"{path} is expected to contain sparse raw counts")
            labels = adata.obs["target_gene"].astype(str).to_numpy()
            batches = adata.obs.get(
                "batch", pd.Series("all", index=adata.obs_names)
            ).astype(str).to_numpy()
            guides = adata.obs.get(
                "guide_id", pd.Series("all", index=adata.obs_names)
            ).astype(str).to_numpy()
            source_genes = gene_names(adata)
            source_columns: list[int] = []
            target_columns: list[int] = []
            for source_index, gene in enumerate(source_genes):
                target_index = gene_lookup.get(gene)
                if target_index is not None:
                    source_columns.append(source_index)
                    target_columns.append(target_index)
            source_index_array = np.asarray(source_columns, dtype=np.int64)
            target_index_array = np.asarray(target_columns, dtype=np.int64)
            matrix = log_cp10k(adata.X)

            control_rows = np.flatnonzero(labels == CONTROL_LABEL)
            fallback_control = aligned_mean(
                matrix,
                control_rows,
                source_index_array,
                target_index_array,
                n_genes,
            )
            control_by_batch: dict[str, np.ndarray] = {}
            for batch in np.unique(batches[control_rows]):
                rows = control_rows[batches[control_rows] == batch]
                if len(rows):
                    control_by_batch[str(batch)] = aligned_mean(
                        matrix,
                        rows,
                        source_index_array,
                        target_index_array,
                        n_genes,
                    )

            for target in sorted(set(labels) - {CONTROL_LABEL}):
                if target not in gene_lookup:
                    continue
                rows = np.flatnonzero(labels == target)
                if len(rows) < args.min_cells:
                    continue
                guide_effects: list[np.ndarray] = []
                guide_controls: list[np.ndarray] = []
                guide_sizes: list[int] = []
                for guide in sorted(np.unique(guides[rows])):
                    guide_rows = rows[guides[rows] == guide]
                    if len(guide_rows) < args.min_guide_cells:
                        continue
                    control = matched_control(
                        guide_rows, batches, control_by_batch, fallback_control
                    )
                    observed = aligned_mean(
                        matrix,
                        guide_rows,
                        source_index_array,
                        target_index_array,
                        n_genes,
                    )
                    guide_controls.append(control)
                    guide_effects.append(observed - control)
                    guide_sizes.append(len(guide_rows))
                if not guide_effects:
                    control = matched_control(
                        rows, batches, control_by_batch, fallback_control
                    )
                    observed = aligned_mean(
                        matrix,
                        rows,
                        source_index_array,
                        target_index_array,
                        n_genes,
                    )
                    guide_controls = [control]
                    guide_effects = [observed - control]
                    guide_sizes = [len(rows)]

                target_index = len(target_names)
                target_names.append(target)
                target_controls.append(np.mean(guide_controls, axis=0))
                target_effects.append(np.mean(guide_effects, axis=0))
                target_cells.append(len(rows))
                target_guides.append(len(guide_effects))
                target_sources.append(path.name)
                for control, effect, size in zip(
                    guide_controls, guide_effects, guide_sizes
                ):
                    replicate_controls.append(control)
                    replicate_effects.append(effect)
                    replicate_targets.append(target_index)
                    replicate_cells.append(size)
        finally:
            del adata

    targets = np.asarray(target_names, dtype=str)
    controls = np.asarray(target_controls, dtype=np.float32)
    effects = np.asarray(target_effects, dtype=np.float32)
    replicate_target_array = np.asarray(replicate_targets, dtype=np.int64)
    self_effect = np.asarray(
        [effects[i, gene_lookup[target]] for i, target in enumerate(targets)]
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        genes=np.asarray(genes, dtype=str),
        targets=targets,
        controls=controls,
        effects=effects,
        target_cells=np.asarray(target_cells, dtype=np.int64),
        target_guides=np.asarray(target_guides, dtype=np.int64),
        target_sources=np.asarray(target_sources, dtype=str),
        replicate_controls=np.asarray(replicate_controls, dtype=np.float32),
        replicate_effects=np.asarray(replicate_effects, dtype=np.float32),
        replicate_targets=replicate_target_array,
        replicate_cells=np.asarray(replicate_cells, dtype=np.int64),
    )
    metadata = {
        "genes": len(genes),
        "targets": len(targets),
        "replicates": len(replicate_target_array),
        "cells": {
            "min": int(np.min(target_cells)),
            "median": float(np.median(target_cells)),
            "max": int(np.max(target_cells)),
        },
        "guides": {
            "min": int(np.min(target_guides)),
            "median": float(np.median(target_guides)),
            "max": int(np.max(target_guides)),
        },
        "self_effect": {
            "negative_fraction": float(np.mean(self_effect < 0)),
            "median": float(np.median(self_effect)),
            "p05": float(np.quantile(self_effect, 0.05)),
            "p95": float(np.quantile(self_effect, 0.95)),
        },
        "effect_abs": {
            "median": float(np.median(np.abs(effects))),
            "p95": float(np.quantile(np.abs(effects), 0.95)),
            "p99": float(np.quantile(np.abs(effects), 0.99)),
            "max": float(np.max(np.abs(effects))),
        },
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
