"""Build aligned pseudobulk signatures from the normalized Replogle K562 file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--h5ad", required=True)
    p.add_argument("--genes", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--condition-col", default="condition")
    args = p.parse_args()
    genes = pd.read_csv(args.genes, header=None).iloc[:, 0].astype(str).tolist()
    lookup = {g: i for i, g in enumerate(genes)}
    a = ad.read_h5ad(args.h5ad)
    if args.condition_col not in a.obs:
        raise ValueError(f"missing {args.condition_col!r} in obs")
    source_genes = a.var["gene_name"].astype(str).tolist() if "gene_name" in a.var else a.var_names.astype(str).tolist()
    source_cols = np.asarray([i for i, g in enumerate(source_genes) if g in lookup], dtype=np.int64)
    target_cols = np.asarray([lookup[source_genes[i]] for i in source_cols], dtype=np.int64)
    cond = a.obs[args.condition_col].astype(str).to_numpy()
    control_rows = np.flatnonzero(cond == "ctrl")
    if len(control_rows) < 20:
        raise ValueError("K562 file has too few ctrl cells")
    matrix = a.X
    def mean_rows(rows: np.ndarray) -> np.ndarray:
        values = matrix[rows][:, source_cols]
        out = np.asarray(values.mean(axis=0)).ravel().astype(np.float32)
        aligned = np.zeros(len(genes), dtype=np.float32)
        aligned[target_cols] = out
        return aligned
    fallback = mean_rows(control_rows)
    targets, controls, effects, cells = [], [], [], []
    replicate_controls, replicate_effects, replicate_targets, replicate_cells = [], [], [], []
    for c in sorted(set(cond) - {"ctrl"}):
        if not c.endswith("+ctrl"):
            continue
        target = c[:-5]
        if target not in lookup:
            continue
        rows = np.flatnonzero(cond == c)
        if len(rows) < 20:
            continue
        observed = mean_rows(rows)
        effect = observed - fallback
        index = len(targets)
        targets.append(target); controls.append(fallback); effects.append(effect); cells.append(len(rows))
        replicate_controls.append(fallback); replicate_effects.append(effect); replicate_targets.append(index); replicate_cells.append(len(rows))
    if not targets:
        raise ValueError("no compatible single-gene K562 conditions found")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, genes=np.asarray(genes, dtype=str), targets=np.asarray(targets, dtype=str),
        controls=np.asarray(controls, dtype=np.float32), effects=np.asarray(effects, dtype=np.float32),
        target_cells=np.asarray(cells, dtype=np.int64), target_guides=np.ones(len(targets), dtype=np.int64),
        target_sources=np.full(len(targets), Path(args.h5ad).name, dtype=str),
        replicate_controls=np.asarray(replicate_controls, dtype=np.float32), replicate_effects=np.asarray(replicate_effects, dtype=np.float32),
        replicate_targets=np.asarray(replicate_targets, dtype=np.int64), replicate_cells=np.asarray(replicate_cells, dtype=np.float32))
    out.with_suffix(".json").write_text(json.dumps({"source": args.h5ad, "normalized": True, "targets": len(targets), "control_cells": len(control_rows)}, indent=2) + "\n")
    print(json.dumps({"targets": len(targets), "control_cells": len(control_rows), "genes_mapped": len(source_cols)}), flush=True)


if __name__ == "__main__":
    main()
