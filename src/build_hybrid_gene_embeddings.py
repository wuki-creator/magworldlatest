"""Combine VCC control co-expression with pretrained scGPT gene features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--genes", required=True)
    parser.add_argument("--coexpression", required=True)
    parser.add_argument("--scgpt-model", required=True)
    parser.add_argument("--scgpt-vocab", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scgpt-components", type=int, default=128)
    parser.add_argument("--seed", type=int, default=127)
    return parser.parse_args()


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def main() -> None:
    args = parse_args()
    genes = pd.read_csv(args.genes).iloc[:, 0].astype(str).tolist()
    coexpression = np.load(args.coexpression).astype(np.float32)
    if coexpression.shape[0] != len(genes):
        raise ValueError("co-expression rows must match the official gene list")

    vocab = json.loads(Path(args.scgpt_vocab).read_text(encoding="utf-8"))
    state = torch.load(args.scgpt_model, map_location="cpu", weights_only=False)
    pretrained = state["encoder.embedding.weight"].float().numpy()
    scgpt = np.zeros((len(genes), pretrained.shape[1]), dtype=np.float32)
    covered = np.zeros(len(genes), dtype=bool)
    for row, gene in enumerate(genes):
        index = vocab.get(gene)
        if index is not None:
            scgpt[row] = pretrained[int(index)]
            covered[row] = True
    if covered.sum() < len(genes) // 2:
        raise ValueError("scGPT vocabulary coverage is unexpectedly low")
    scgpt[covered] = normalize_rows(scgpt[covered])
    svd = TruncatedSVD(
        n_components=args.scgpt_components,
        n_iter=7,
        random_state=args.seed,
    )
    reduced_covered = svd.fit_transform(scgpt[covered]).astype(np.float32)
    reduced = np.zeros((len(genes), args.scgpt_components), dtype=np.float32)
    reduced[covered] = normalize_rows(reduced_covered)

    coexpression = normalize_rows(coexpression)
    hybrid = normalize_rows(np.concatenate([coexpression, reduced], axis=1)).astype(
        np.float32
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, hybrid)
    metadata = {
        "shape": list(hybrid.shape),
        "scgpt_vocab": len(vocab),
        "covered_genes": int(covered.sum()),
        "missing_genes": int((~covered).sum()),
        "scgpt_components": args.scgpt_components,
        "scgpt_explained_variance_ratio": float(
            svd.explained_variance_ratio_.sum()
        ),
        "seed": args.seed,
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
