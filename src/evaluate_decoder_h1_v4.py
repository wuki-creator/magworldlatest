"""Evaluate VCC decoders on held-out H1 cells with distributional DE metrics."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.stats import mannwhitneyu


CONTROL_LABEL = "non-targeting"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--ensemble-weights", nargs="+", type=float, required=True)
    parser.add_argument("--h1", nargs="+", required=True)
    parser.add_argument("--signatures", required=True)
    parser.add_argument("--genes", required=True)
    parser.add_argument("--perts", required=True)
    parser.add_argument("--magworld-src", required=True)
    parser.add_argument("--model-module", default="model_world_h1_v3")
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-k", nargs="+", type=int, default=(100, 200, 500))
    parser.add_argument("--downstream-scale", nargs="+", type=float, default=(0.25, 0.5, 1.0))
    parser.add_argument("--prior-strength", nargs="+", type=float, default=(0.5, 2.0))
    parser.add_argument("--self-scale", type=float, default=1.38)
    parser.add_argument("--max-delta", type=float, default=2.0)
    parser.add_argument("--cells", type=int, default=400)
    parser.add_argument("--candidate-genes", type=int, default=1000)
    parser.add_argument("--fdr", type=float, default=0.05)
    parser.add_argument("--min-effect", type=float, default=0.05)
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def bh_adjust_subset(p_values: np.ndarray, total_tests: int) -> np.ndarray:
    """BH-adjust a tested subset while treating untested genes as p=1."""
    order = np.argsort(p_values)
    ranked = p_values[order] * total_tests / np.arange(1, len(p_values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1].clip(0.0, 1.0)
    adjusted = np.empty_like(ranked)
    adjusted[order] = ranked
    return adjusted


def log_cp10k(matrix: sp.csr_matrix, columns: np.ndarray) -> np.ndarray:
    library = np.asarray(matrix.sum(1)).ravel().astype(np.float64)
    selected = matrix[:, columns].astype(np.float64)
    selected = selected.multiply((10_000.0 / np.maximum(library, 1.0))[:, None])
    dense = selected.toarray()
    np.log1p(dense, out=dense)
    return dense


def de_statistics(
    treated: sp.csr_matrix,
    control: sp.csr_matrix,
    columns: np.ndarray,
    total_genes: int,
) -> tuple[np.ndarray, np.ndarray]:
    treated_log = log_cp10k(treated, columns)
    control_log = log_cp10k(control, columns)
    effect = treated_log.mean(0) - control_log.mean(0)
    result = mannwhitneyu(
        treated_log,
        control_log,
        axis=0,
        alternative="two-sided",
        method="asymptotic",
        use_continuity=True,
    )
    p_values = np.nan_to_num(result.pvalue, nan=1.0, posinf=1.0, neginf=1.0)
    return effect.astype(np.float32), bh_adjust_subset(p_values, total_genes)


def select_rows(rows: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(rows, n, replace=len(rows) < n)


def matched_control_rows(
    target_rows: np.ndarray,
    batches: np.ndarray,
    control_rows: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    selected_target = select_rows(target_rows, n, rng)
    target_batches = batches[selected_target]
    selected: list[int] = []
    fallback = control_rows
    for batch in np.unique(target_batches):
        take = int(np.sum(target_batches == batch))
        candidates = control_rows[batches[control_rows] == batch]
        if len(candidates) == 0:
            candidates = fallback
        selected.extend(rng.choice(candidates, take, replace=len(candidates) < take).tolist())
    return selected_target, np.asarray(selected, dtype=np.int64)


def metrics(
    predicted_effect: np.ndarray,
    predicted_q: np.ndarray,
    truth_effect: np.ndarray,
    truth_q: np.ndarray,
    fdr: float,
    min_effect: float,
) -> dict[str, float]:
    truth_mask = (truth_q <= fdr) & (np.abs(truth_effect) >= min_effect)
    predicted_mask = (predicted_q <= fdr) & (np.abs(predicted_effect) >= min_effect)
    truth_set = set(np.flatnonzero(truth_mask).tolist())
    predicted_set = set(np.flatnonzero(predicted_mask).tolist())
    overlap = np.asarray(sorted(truth_set & predicted_set), dtype=np.int64)
    union = truth_set | predicted_set
    sign_matches = int(
        np.sum(np.sign(predicted_effect[overlap]) == np.sign(truth_effect[overlap]))
    ) if len(overlap) else 0
    truth_norm = float(np.sum(np.abs(truth_effect[truth_mask])))
    pred_norm = float(np.linalg.norm(predicted_effect))
    true_l2 = float(np.linalg.norm(truth_effect))
    return {
        "truth_de": float(len(truth_set)),
        "predicted_de": float(len(predicted_set)),
        "direction_fidelity": sign_matches / max(1, len(overlap)),
        "signed_reach": sign_matches / max(1, len(truth_set)),
        "reach": len(overlap) / max(1, len(truth_set)),
        "jaccard": len(overlap) / max(1, len(union)),
        "lfc_nmae": float(
            np.sum(np.abs(predicted_effect[truth_mask] - truth_effect[truth_mask]))
            / max(truth_norm, 1e-8)
        ),
        "cosine": float(
            np.dot(predicted_effect, truth_effect) / max(pred_norm * true_l2, 1e-8)
        ),
    }


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def main() -> None:
    args = parse_args()
    started = time.time()
    module_path = str(Path(args.magworld_src).resolve())
    if module_path not in sys.path:
        sys.path.insert(0, module_path)
    from predict_magworld_vcc2026_v3 import (  # noqa: PLC0415
        adata_genes,
        aligned_sparse,
        bayesian_decode,
        calibrate_effect,
        load_models,
        predict_effects,
        read_genes,
    )
    model_class = importlib.import_module(args.model_module).WorldModelH1V3

    genes = read_genes(args.genes)
    gene_lookup = {gene: index for index, gene in enumerate(genes)}
    official = set(pd.read_csv(args.perts)["target_gene"].astype(str))
    signatures = np.load(args.signatures, allow_pickle=False)
    signature_targets = signatures["targets"].astype(str)
    signature_sources = signatures["target_sources"].astype(str)
    signature_effects = signatures["effects"].astype(np.float32)
    eligible = [
        index for index, target in enumerate(signature_targets)
        if target in official and target in gene_lookup
    ]
    if args.max_targets is not None:
        eligible = eligible[: args.max_targets]
    by_source: dict[str, list[int]] = defaultdict(list)
    for index in eligible:
        by_source[signature_sources[index]].append(index)

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    models = load_models(args.checkpoints, model_class, genes, device)
    weights = np.asarray(args.ensemble_weights, dtype=np.float64)
    if len(weights) != len(models) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("invalid ensemble weights")
    weights /= weights.sum()

    configurations = [
        (top_k, scale, prior)
        for top_k in args.top_k
        for scale in args.downstream_scale
        for prior in args.prior_strength
    ]
    results: dict[str, list[dict[str, float]]] = {
        f"k={top_k},scale={scale:g},prior={prior:g}": []
        for top_k, scale, prior in configurations
    }
    per_target: list[dict[str, object]] = []
    h1_lookup = {Path(path).name: path for path in args.h1}

    for source_name, target_positions in sorted(by_source.items()):
        source_path = h1_lookup.get(source_name)
        if source_path is None:
            raise ValueError(f"missing H1 source {source_name}")
        adata = ad.read_h5ad(source_path, backed="r")
        try:
            labels = adata.obs["target_gene"].astype(str).to_numpy()
            batches = adata.obs["batch"].astype(str).to_numpy()
            control_rows = np.flatnonzero(labels == CONTROL_LABEL)
            source_genes = adata_genes(adata)
            for signature_index in target_positions:
                target = signature_targets[signature_index]
                target_rows = np.flatnonzero(labels == target)
                rng = np.random.default_rng(args.seed + signature_index)
                selected_target, selected_control = matched_control_rows(
                    target_rows, batches, control_rows, args.cells, rng
                )
                base = aligned_sparse(
                    adata.X[np.sort(selected_control)], source_genes, genes
                )
                truth = aligned_sparse(
                    adata.X[np.sort(selected_target)], source_genes, genes
                )
                normalized = base.multiply(
                    (10_000.0 / np.maximum(np.asarray(base.sum(1)).ravel(), 1.0))[:, None]
                ).tocsr()
                normalized.data = np.log1p(normalized.data)
                control_mean = np.asarray(normalized.mean(0)).ravel().astype(np.float32)
                target_index = gene_lookup[target]
                raw_delta = predict_effects(
                    models,
                    weights,
                    control_mean,
                    np.asarray([target_index], dtype=np.int64),
                    1,
                    device,
                )[0]
                truth_hint = signature_effects[signature_index]
                candidate_count = min(args.candidate_genes, len(genes) - 1)
                predicted_candidates = np.argpartition(
                    np.abs(raw_delta), -candidate_count
                )[-candidate_count:]
                truth_candidates = np.argpartition(
                    np.abs(truth_hint), -candidate_count
                )[-candidate_count:]
                candidates = np.unique(
                    np.concatenate(
                        [predicted_candidates, truth_candidates, [target_index]]
                    )
                )
                truth_effect, truth_q = de_statistics(
                    truth, base, candidates, len(genes)
                )
                gene_mean = np.asarray(base.mean(0)).ravel().astype(np.float64)
                target_summary: dict[str, object] = {"target": target, "source": source_name}
                for config_index, (top_k, scale, prior) in enumerate(configurations):
                    delta = calibrate_effect(
                        raw_delta,
                        target_index,
                        top_k,
                        scale,
                        args.self_scale,
                        args.max_delta,
                    )
                    decoded = bayesian_decode(
                        base,
                        delta,
                        gene_mean,
                        np.random.default_rng(
                            args.seed + 100_000 * signature_index + config_index
                        ),
                        prior,
                    )
                    predicted_effect, predicted_q = de_statistics(
                        decoded, base, candidates, len(genes)
                    )
                    row = metrics(
                        predicted_effect,
                        predicted_q,
                        truth_effect,
                        truth_q,
                        args.fdr,
                        args.min_effect,
                    )
                    key = f"k={top_k},scale={scale:g},prior={prior:g}"
                    results[key].append(row)
                best_key = max(
                    results,
                    key=lambda key: (
                        results[key][-1]["signed_reach"]
                        if len(results[key]) == len(per_target) + 1
                        else -math.inf
                    ),
                )
                target_summary["best_signed_reach_config"] = best_key
                per_target.append(target_summary)
                print(
                    json.dumps(
                        {"completed": len(per_target), "total": len(eligible), "target": target}
                    ),
                    flush=True,
                )
        finally:
            adata.file.close()

    aggregated = {key: aggregate(rows) for key, rows in results.items()}
    ranking = sorted(
        aggregated,
        key=lambda key: (
            aggregated[key]["signed_reach"],
            aggregated[key]["jaccard"],
            aggregated[key]["cosine"],
            -aggregated[key]["lfc_nmae"],
        ),
        reverse=True,
    )
    payload = {
        "targets": len(eligible),
        "best": ranking[0],
        "ranking": ranking,
        "metrics": aggregated,
        "per_target": per_target,
        "seconds": round(time.time() - started, 2),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
