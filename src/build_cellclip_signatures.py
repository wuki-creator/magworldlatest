"""Build batch-matched signatures from CellClip/scPerturb training units."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

from build_h1_signature_dataset import (
    aligned_mean,
    gene_names,
    log_cp10k,
    matched_control,
    read_genes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--genes", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-cells", type=int, default=24)
    parser.add_argument("--min-guide-cells", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.h5ad)
    genes = read_genes(args.genes)
    lookup = {gene: index for index, gene in enumerate(genes)}
    adata = ad.read_h5ad(path)
    try:
        required_obs = {
            "target_gene_symbol",
            "target_gene_symbol_status",
            "is_reference",
            "raw_guide_id",
            "batch",
            "perturbation_method",
        }
        missing = sorted(required_obs - set(adata.obs.columns))
        if missing:
            raise ValueError(f"missing CellClip obs columns: {missing}")
        if "counts" not in adata.layers:
            raise ValueError("CellClip unit does not contain layers['counts']")
        counts = adata.layers["counts"]
        if sp.issparse(counts):
            counts = sp.csr_matrix(counts, dtype=np.float32)
        else:
            dense_counts = np.asarray(counts)
            if not np.all(np.isfinite(dense_counts)) or np.any(dense_counts < 0):
                raise ValueError("CellClip dense counts contain invalid values")
            counts = sp.csr_matrix(dense_counts, dtype=np.float32)
            del dense_counts
        if counts.data.size and not np.all(counts.data == np.floor(counts.data)):
            raise ValueError("CellClip counts layer contains non-integer values")

        labels = adata.obs["target_gene_symbol"].fillna("").astype(str).to_numpy()
        statuses = adata.obs["target_gene_symbol_status"].astype(str).to_numpy()
        reference = adata.obs["is_reference"].astype(bool).to_numpy()
        guides = adata.obs["raw_guide_id"].astype(str).to_numpy()
        batches = adata.obs["batch"].astype(str).to_numpy()
        methods = sorted(adata.obs["perturbation_method"].astype(str).unique())

        source_genes = gene_names(adata)
        source_columns = np.asarray(
            [index for index, gene in enumerate(source_genes) if gene in lookup],
            dtype=np.int64,
        )
        target_columns = np.asarray(
            [lookup[source_genes[index]] for index in source_columns],
            dtype=np.int64,
        )
        matrix = log_cp10k(counts)
        control_rows = np.flatnonzero(reference)
        if len(control_rows) < args.min_cells:
            raise ValueError("too few authoritative reference cells")
        fallback = aligned_mean(
            matrix, control_rows, source_columns, target_columns, len(genes)
        )
        by_batch = {
            batch: aligned_mean(
                matrix,
                control_rows[batches[control_rows] == batch],
                source_columns,
                target_columns,
                len(genes),
            )
            for batch in np.unique(batches[control_rows])
            if np.any(batches[control_rows] == batch)
        }

        target_names: list[str] = []
        controls: list[np.ndarray] = []
        effects: list[np.ndarray] = []
        target_cells: list[int] = []
        target_guides: list[int] = []
        replicate_controls: list[np.ndarray] = []
        replicate_effects: list[np.ndarray] = []
        replicate_targets: list[int] = []
        replicate_cells: list[int] = []
        eligible = (~reference) & (statuses == "mapped_to_foundation_vocab")
        for target in sorted(set(labels[eligible])):
            if not target or target not in lookup:
                continue
            rows = np.flatnonzero(eligible & (labels == target))
            valid_guides = [
                guide
                for guide in np.unique(guides[rows])
                if np.sum(guides[rows] == guide) >= args.min_guide_cells
            ]
            if len(rows) < args.min_cells or not valid_guides:
                continue
            target_index = len(target_names)
            guide_controls: list[np.ndarray] = []
            guide_effects: list[np.ndarray] = []
            for guide in valid_guides:
                guide_rows = rows[guides[rows] == guide]
                control = matched_control(guide_rows, batches, by_batch, fallback)
                observed = aligned_mean(
                    matrix,
                    guide_rows,
                    source_columns,
                    target_columns,
                    len(genes),
                )
                effect = observed - control
                guide_controls.append(control)
                guide_effects.append(effect)
                replicate_controls.append(control)
                replicate_effects.append(effect)
                replicate_targets.append(target_index)
                replicate_cells.append(len(guide_rows))
            target_names.append(target)
            controls.append(np.mean(guide_controls, axis=0))
            effects.append(np.mean(guide_effects, axis=0))
            target_cells.append(len(rows))
            target_guides.append(len(valid_guides))

        if not target_names:
            raise ValueError("no compatible perturbation targets found")
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "source": str(path),
            "source_collection": "CellClip scPerturb-derived release",
            "license": "CC BY 4.0",
            "perturbation_methods": methods,
            "transformation": "layers[counts] -> log1p(CP10K)",
            "control_selector": "obs.is_reference == True",
            "targets": len(target_names),
            "replicates": len(replicate_targets),
            "reference_cells": len(control_rows),
            "mapped_genes": len(source_columns),
        }
        np.savez_compressed(
            output,
            genes=np.asarray(genes, dtype=str),
            targets=np.asarray(target_names, dtype=str),
            controls=np.asarray(controls, dtype=np.float32),
            effects=np.asarray(effects, dtype=np.float32),
            target_cells=np.asarray(target_cells, dtype=np.int64),
            target_guides=np.asarray(target_guides, dtype=np.int64),
            target_sources=np.full(len(target_names), path.name, dtype=str),
            replicate_controls=np.asarray(replicate_controls, dtype=np.float32),
            replicate_effects=np.asarray(replicate_effects, dtype=np.float32),
            replicate_targets=np.asarray(replicate_targets, dtype=np.int64),
            replicate_cells=np.asarray(replicate_cells, dtype=np.int64),
        )
        output.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(metadata), flush=True)
    finally:
        del adata


if __name__ == "__main__":
    main()
