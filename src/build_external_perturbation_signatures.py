"""Convert compatible external single-cell CRISPRi screens into H1 signatures.

The input contract intentionally requires raw sparse counts plus target, guide,
batch, and control labels. Incompatible screens are rejected rather than mixed
into the VCC training domain without provenance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from build_h1_signature_dataset import CONTROL_LABEL, aligned_mean, gene_names, log_cp10k, matched_control, read_genes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h1", nargs="+", required=True)
    parser.add_argument("--genes", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-col", default="target_gene")
    parser.add_argument("--guide-col", default="guide_id")
    parser.add_argument("--batch-col", default="batch")
    parser.add_argument("--control-label", default=CONTROL_LABEL)
    parser.add_argument("--min-cells", type=int, default=24)
    parser.add_argument("--min-guide-cells", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    genes = read_genes(args.genes)
    lookup = {gene: index for index, gene in enumerate(genes)}
    target_names: list[str] = []
    controls: list[np.ndarray] = []
    effects: list[np.ndarray] = []
    target_cells: list[int] = []
    target_guides: list[int] = []
    sources: list[str] = []
    replicate_controls: list[np.ndarray] = []
    replicate_effects: list[np.ndarray] = []
    replicate_targets: list[int] = []
    replicate_cells: list[int] = []
    rejected: list[dict[str, object]] = []
    for h1_path in args.h1:
        path = Path(h1_path)
        try:
            adata = ad.read_h5ad(path)
            if not sp.issparse(adata.X):
                raise ValueError("X is not sparse raw counts")
            missing = [column for column in (args.target_col, args.guide_col, args.batch_col) if column not in adata.obs]
            if missing:
                raise ValueError(f"missing obs columns: {missing}")
            labels = adata.obs[args.target_col].astype(str).to_numpy()
            guides = adata.obs[args.guide_col].astype(str).to_numpy()
            batches = adata.obs[args.batch_col].astype(str).to_numpy()
            source_genes = gene_names(adata)
            source_cols = np.asarray([i for i, gene in enumerate(source_genes) if gene in lookup], dtype=np.int64)
            target_cols = np.asarray([lookup[source_genes[i]] for i in source_cols], dtype=np.int64)
            matrix = log_cp10k(adata.X)
            control_rows = np.flatnonzero(labels == args.control_label)
            if len(control_rows) < args.min_cells:
                raise ValueError("too few control cells")
            fallback = aligned_mean(matrix, control_rows, source_cols, target_cols, len(genes))
            by_batch = {
                batch: aligned_mean(matrix, control_rows[batches[control_rows] == batch], source_cols, target_cols, len(genes))
                for batch in np.unique(batches[control_rows])
            }
            for target in sorted(set(labels) - {args.control_label}):
                if target not in lookup:
                    continue
                rows = np.flatnonzero(labels == target)
                valid_guides = [guide for guide in np.unique(guides[rows]) if np.sum(guides[rows] == guide) >= args.min_guide_cells]
                if len(rows) < args.min_cells or not valid_guides:
                    continue
                target_index = len(target_names)
                guide_controls: list[np.ndarray] = []
                guide_effects: list[np.ndarray] = []
                for guide in valid_guides:
                    guide_rows = rows[guides[rows] == guide]
                    control = matched_control(guide_rows, batches, by_batch, fallback)
                    observed = aligned_mean(matrix, guide_rows, source_cols, target_cols, len(genes))
                    guide_controls.append(control)
                    guide_effects.append(observed - control)
                    replicate_controls.append(control)
                    replicate_effects.append(observed - control)
                    replicate_targets.append(target_index)
                    replicate_cells.append(len(guide_rows))
                target_names.append(target)
                controls.append(np.mean(guide_controls, axis=0))
                effects.append(np.mean(guide_effects, axis=0))
                sources.append(path.name)
                target_cells.append(len(rows))
                target_guides.append(len(valid_guides))
        except Exception as exc:
            rejected.append({"source": str(path), "reason": str(exc)})
        finally:
            if "adata" in locals():
                del adata
    if not target_names:
        raise ValueError(f"no compatible external targets; rejected={rejected}")
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        genes=np.asarray(genes, dtype=str),
        targets=np.asarray(target_names, dtype=str),
        controls=np.asarray(controls, dtype=np.float32),
        effects=np.asarray(effects, dtype=np.float32),
        target_cells=np.asarray(target_cells, dtype=np.int64),
        target_guides=np.asarray(target_guides, dtype=np.int64),
        target_sources=np.asarray(sources, dtype=str),
        replicate_controls=np.asarray(replicate_controls, dtype=np.float32),
        replicate_effects=np.asarray(replicate_effects, dtype=np.float32),
        replicate_targets=np.asarray(replicate_targets, dtype=np.int64),
        replicate_cells=np.asarray(replicate_cells, dtype=np.int64),
    )
    output.with_suffix(".json").write_text(json.dumps({"targets": len(target_names), "rejected": rejected}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"targets": len(target_names), "rejected": rejected}), flush=True)


if __name__ == "__main__":
    main()
