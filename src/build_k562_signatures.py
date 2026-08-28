"""Build batch-matched signatures from the Replogle K562 raw-count file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--genes", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-col", default="gene")
    parser.add_argument("--guide-col", default="sgID_AB")
    parser.add_argument("--batch-col", default="gem_group")
    parser.add_argument("--control-label", default="non-targeting")
    parser.add_argument("--min-cells", type=int, default=20)
    parser.add_argument("--min-guide-cells", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gene_frame = pd.read_csv(args.genes)
    gene_column = "gene_name" if "gene_name" in gene_frame else gene_frame.columns[0]
    genes = gene_frame[gene_column].astype(str).tolist()
    gene_lookup = {gene: index for index, gene in enumerate(genes)}
    adata = ad.read_h5ad(args.h5ad, backed="r")
    required = (args.target_col, args.guide_col, args.batch_col)
    missing = [column for column in required if column not in adata.obs]
    if missing:
        raise ValueError(f"missing obs columns: {missing}")

    labels = adata.obs[args.target_col].astype(str).to_numpy()
    guides = adata.obs[args.guide_col].astype(str).to_numpy()
    batches = adata.obs[args.batch_col].astype(str).to_numpy()
    source_genes = (
        adata.var["gene_name"].astype(str).tolist()
        if "gene_name" in adata.var
        else adata.var_names.astype(str).tolist()
    )
    source_cols = np.asarray(
        [index for index, gene in enumerate(source_genes) if gene in gene_lookup],
        dtype=np.int64,
    )
    target_cols = np.asarray(
        [gene_lookup[source_genes[index]] for index in source_cols], dtype=np.int64
    )
    batch_names, batch_codes = np.unique(batches, return_inverse=True)
    guide_names, guide_codes = np.unique(guides, return_inverse=True)
    guide_targets = np.empty(len(guide_names), dtype=object)
    for code in range(len(guide_names)):
        values, counts = np.unique(labels[guide_codes == code], return_counts=True)
        guide_targets[code] = values[np.argmax(counts)]

    n_source = len(source_cols)
    control_sums = np.zeros((len(batch_names), n_source), dtype=np.float64)
    control_counts = np.zeros(len(batch_names), dtype=np.int64)
    guide_sums = np.zeros((len(guide_names), n_source), dtype=np.float64)
    guide_counts = np.zeros(len(guide_names), dtype=np.int64)
    guide_batch_counts = np.zeros(
        (len(guide_names), len(batch_names)), dtype=np.int64
    )

    for start in range(0, adata.n_obs, args.chunk_size):
        end = min(adata.n_obs, start + args.chunk_size)
        values = np.asarray(adata.X[start:end, source_cols], dtype=np.float32)
        library = values.sum(axis=1, keepdims=True)
        values = np.log1p(values * (10_000.0 / np.maximum(library, 1.0)))
        local_labels = labels[start:end]
        local_guides = guide_codes[start:end]
        local_batches = batch_codes[start:end]
        control_mask = local_labels == args.control_label
        for batch in np.unique(local_batches[control_mask]):
            mask = control_mask & (local_batches == batch)
            control_sums[batch] += values[mask].sum(axis=0, dtype=np.float64)
            control_counts[batch] += int(mask.sum())
        for guide in np.unique(local_guides):
            mask = local_guides == guide
            guide_sums[guide] += values[mask].sum(axis=0, dtype=np.float64)
            guide_counts[guide] += int(mask.sum())
            np.add.at(guide_batch_counts[guide], local_batches[mask], 1)
        if end % (20 * args.chunk_size) < args.chunk_size:
            print(f"cells={end}/{adata.n_obs}", flush=True)

    fallback = control_sums.sum(axis=0) / max(1, int(control_counts.sum()))
    batch_controls = np.repeat(fallback[None, :], len(batch_names), axis=0)
    valid_batches = control_counts > 0
    batch_controls[valid_batches] = (
        control_sums[valid_batches] / control_counts[valid_batches, None]
    )
    target_to_guides: dict[str, list[int]] = {}
    for code, target in enumerate(guide_targets.astype(str)):
        if target != args.control_label and target in gene_lookup and guide_counts[code] >= args.min_guide_cells:
            target_to_guides.setdefault(target, []).append(code)

    targets: list[str] = []
    controls: list[np.ndarray] = []
    effects: list[np.ndarray] = []
    target_cells: list[int] = []
    target_guides: list[int] = []
    replicate_controls: list[np.ndarray] = []
    replicate_effects: list[np.ndarray] = []
    replicate_targets: list[int] = []
    replicate_cells: list[int] = []
    for target in sorted(target_to_guides):
        codes = target_to_guides[target]
        if sum(int(guide_counts[code]) for code in codes) < args.min_cells:
            continue
        target_index = len(targets)
        local_controls: list[np.ndarray] = []
        local_effects: list[np.ndarray] = []
        for code in codes:
            weights = guide_batch_counts[code].astype(np.float64)
            control = weights @ batch_controls / max(1.0, weights.sum())
            observed = guide_sums[code] / max(1, int(guide_counts[code]))
            effect = observed - control
            aligned_control = np.zeros(len(genes), dtype=np.float32)
            aligned_effect = np.zeros(len(genes), dtype=np.float32)
            aligned_control[target_cols] = control.astype(np.float32)
            aligned_effect[target_cols] = effect.astype(np.float32)
            local_controls.append(aligned_control)
            local_effects.append(aligned_effect)
            replicate_controls.append(aligned_control)
            replicate_effects.append(aligned_effect)
            replicate_targets.append(target_index)
            replicate_cells.append(int(guide_counts[code]))
        targets.append(target)
        controls.append(np.mean(local_controls, axis=0))
        effects.append(np.mean(local_effects, axis=0))
        target_cells.append(sum(int(guide_counts[code]) for code in codes))
        target_guides.append(len(codes))

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        genes=np.asarray(genes, dtype=str), targets=np.asarray(targets, dtype=str),
        controls=np.asarray(controls, dtype=np.float32), effects=np.asarray(effects, dtype=np.float32),
        target_cells=np.asarray(target_cells, dtype=np.int64), target_guides=np.asarray(target_guides, dtype=np.int64),
        target_sources=np.full(len(targets), Path(args.h5ad).name, dtype=str),
        replicate_controls=np.asarray(replicate_controls, dtype=np.float32),
        replicate_effects=np.asarray(replicate_effects, dtype=np.float32),
        replicate_targets=np.asarray(replicate_targets, dtype=np.int64),
        replicate_cells=np.asarray(replicate_cells, dtype=np.float32),
    )
    metadata = {
        "source": args.h5ad, "raw_integer_counts": True,
        "transform": "log1p(CP10K)", "targets": len(targets),
        "guides": len(replicate_targets), "control_cells": int(control_counts.sum()),
        "genes_mapped": len(source_cols),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
