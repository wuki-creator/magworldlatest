"""Evaluate robust-consensus decoder settings on held-out H1 perturbations."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
from collections import defaultdict
from itertools import product
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from distributional_decoder import (
    build_distribution_reference,
    distribution_metrics,
    heterogeneous_bayesian_decode,
    match_library_sizes,
)
from evaluate_decoder_h1_v4 import (
    CONTROL_LABEL,
    aggregate,
    de_statistics,
    matched_control_rows,
    metrics,
)
from predict_magworld_vcc2026_v4 import (
    adata_genes,
    aligned_sparse,
    bayesian_decode,
    load_models,
    read_genes,
)
from predict_magworld_vcc2026_v5 import (
    center_panel_effects,
    consensus_calibrate_effect,
    consensus_statistics,
    normalize_weights,
    predict_model_effects,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--ensemble-weights", nargs="+", type=float, required=True)
    parser.add_argument("--h1", nargs="+", required=True)
    parser.add_argument("--signatures", required=True)
    parser.add_argument("--genes", required=True)
    parser.add_argument("--perts", required=True)
    parser.add_argument("--magworld-src", required=True)
    parser.add_argument("--model-module", default="model_world_h1_v4")
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-k", nargs="+", type=int, default=(350, 500))
    parser.add_argument(
        "--downstream-scale", nargs="+", type=float, default=(1.25, 1.5, 2.0)
    )
    parser.add_argument("--prior-strength", nargs="+", type=float, default=(2.0,))
    parser.add_argument(
        "--min-sign-agreement", nargs="+", type=float, default=(0.0, 0.5)
    )
    parser.add_argument(
        "--uncertainty-penalty", nargs="+", type=float, default=(0.0, 0.5)
    )
    parser.add_argument(
        "--expression-gate-scale", nargs="+", type=float, default=(0.0, 0.5)
    )
    parser.add_argument(
        "--panel-centering", nargs="+", type=float, default=(0.0, 0.5, 1.0)
    )
    parser.add_argument(
        "--library-match-strength", nargs="+", type=float, default=(0.0,)
    )
    parser.add_argument("--response-sigma", nargs="+", type=float, default=(0.0,))
    parser.add_argument(
        "--strict-all-genes",
        action="store_true",
        help="Run DE over every gene instead of a truth-assisted candidate subset.",
    )
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


def proxy_score(row: dict[str, float]) -> float:
    """Balanced proxy for the six equally weighted 2026 leaderboard metrics."""
    expression_accuracy = 1.0 - min(row["lfc_nmae"], 2.0)
    predicted_de_ratio = row["predicted_de"] / max(row["truth_de"], 1.0)
    yield_match = math.exp(-abs(math.log(max(predicted_de_ratio, 1e-4))))
    return float(
        (
            row["cosine"]
            + expression_accuracy
            + row["direction_fidelity"]
            + row["signed_reach"]
            + row["reach"]
            + row["jaccard"]
        )
        / 6.0
        + 0.05 * yield_match
    )


def strict_proxy_score(row: dict[str, float]) -> float:
    """Map blind offline diagnostics to the six public leaderboard categories."""
    expression_accuracy = 1.0 - min(row["lfc_nmae"], 2.0)
    pds_proxy = (
        row["cosine"] + row["direction_fidelity"] + row["signed_reach"]
    ) / 3.0
    mse_proxy = (row["cosine"] + expression_accuracy) / 2.0
    return float(
        (
            pds_proxy
            + mse_proxy
            + row["jaccard"]
            + expression_accuracy
            + row["latent_fid_similarity"]
            + row["reach"]
        )
        / 6.0
    )


def config_key(
    config: tuple[int, float, float, float, float, float, float, float, float]
) -> str:
    (
        top_k,
        scale,
        prior,
        agreement,
        penalty,
        expression_gate,
        panel_centering,
        library_match,
        response_sigma,
    ) = config
    return (
        f"k={top_k},scale={scale:g},prior={prior:g},agree={agreement:g},"
        f"uncert={penalty:g},expr={expression_gate:g},center={panel_centering:g},"
        f"library={library_match:g},sigma={response_sigma:g}"
    )


def main() -> None:
    args = parse_args()
    started = time.time()
    module_path = str(Path(args.magworld_src).resolve())
    if module_path not in sys.path:
        sys.path.insert(0, module_path)
    model_class = importlib.import_module(args.model_module).WorldModelH1V3

    genes = read_genes(args.genes)
    gene_lookup = {gene: index for index, gene in enumerate(genes)}
    official = set(pd.read_csv(args.perts)["target_gene"].astype(str))
    official_targets = [
        target
        for target in pd.read_csv(args.perts)["target_gene"].astype(str)
        if target in gene_lookup
    ]
    official_target_indices = np.asarray(
        [gene_lookup[target] for target in official_targets], dtype=np.int64
    )
    official_position = {
        target: position for position, target in enumerate(official_targets)
    }
    signatures = np.load(args.signatures, allow_pickle=False)
    signature_targets = signatures["targets"].astype(str)
    signature_sources = signatures["target_sources"].astype(str)
    signature_effects = signatures["effects"].astype(np.float32)
    eligible = [
        index
        for index, target in enumerate(signature_targets)
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
    weights = normalize_weights(np.asarray(args.ensemble_weights), len(models))
    configurations = list(
        product(
            args.top_k,
            args.downstream_scale,
            args.prior_strength,
            args.min_sign_agreement,
            args.uncertainty_penalty,
            args.expression_gate_scale,
            args.panel_centering,
            args.library_match_strength,
            args.response_sigma,
        )
    )
    results: dict[str, list[dict[str, float]]] = {
        config_key(config): [] for config in configurations
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
                    (10_000.0 / np.maximum(np.asarray(base.sum(1)).ravel(), 1.0))[
                        :, None
                    ]
                ).tocsr()
                normalized.data = np.log1p(normalized.data)
                control_mean = np.asarray(normalized.mean(0)).ravel().astype(np.float32)
                target_index = gene_lookup[target]
                panel_model_effects = predict_model_effects(
                    models,
                    control_mean,
                    official_target_indices,
                    32,
                    device,
                )
                panel_position = official_position[target]
                raw_model_effects = panel_model_effects[:, panel_position]
                centered_model_effects = center_panel_effects(
                    panel_model_effects, 1.0
                )[:, panel_position]
                mean_effect, _, _ = consensus_statistics(
                    raw_model_effects, weights, 0.0
                )
                centered_mean_effect, _, _ = consensus_statistics(
                    centered_model_effects, weights, 0.0
                )
                distribution_reference = build_distribution_reference(
                    base, truth, mean_effect
                )
                truth_hint = signature_effects[signature_index]
                candidate_count = min(args.candidate_genes, len(genes) - 1)
                predicted_candidates = np.argpartition(
                    np.abs(mean_effect), -candidate_count
                )[-candidate_count:]
                centered_candidates = np.argpartition(
                    np.abs(centered_mean_effect), -candidate_count
                )[-candidate_count:]
                truth_candidates = np.argpartition(
                    np.abs(truth_hint), -candidate_count
                )[-candidate_count:]
                if args.strict_all_genes:
                    candidates = np.arange(len(genes), dtype=np.int64)
                else:
                    candidates = np.unique(
                        np.concatenate(
                            [
                                predicted_candidates,
                                centered_candidates,
                                truth_candidates,
                                [target_index],
                            ]
                        )
                    )
                truth_effect, truth_q = de_statistics(
                    truth, base, candidates, len(genes)
                )
                gene_mean = np.asarray(base.mean(0)).ravel().astype(np.float64)
                target_rows_metrics: dict[str, dict[str, float]] = {}
                for config in configurations:
                    (
                        top_k,
                        scale,
                        prior,
                        agreement,
                        penalty,
                        expression_gate,
                        panel_centering,
                        library_match,
                        response_sigma,
                    ) = config
                    model_effects = center_panel_effects(
                        panel_model_effects, panel_centering
                    )[:, panel_position]
                    delta, _ = consensus_calibrate_effect(
                        model_effects,
                        weights,
                        gene_mean,
                        target_index,
                        top_k,
                        scale,
                        args.self_scale,
                        args.max_delta,
                        agreement,
                        penalty,
                        expression_gate,
                    )
                    config_rng = np.random.default_rng(
                        args.seed + 100_000 * signature_index
                    )
                    if response_sigma > 0:
                        decoded = heterogeneous_bayesian_decode(
                            base,
                            delta,
                            gene_mean,
                            config_rng,
                            np.random.default_rng(
                                args.seed + 50_000_000 + 100_000 * signature_index
                            ),
                            prior,
                            response_sigma,
                        )
                    else:
                        decoded = bayesian_decode(
                            base,
                            delta,
                            gene_mean,
                            config_rng,
                            prior,
                        )
                    decoded = match_library_sizes(
                        decoded, base, library_match, config_rng
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
                    row.update(distribution_metrics(decoded, distribution_reference))
                    row["proxy_score"] = strict_proxy_score(row)
                    key = config_key(config)
                    results[key].append(row)
                    target_rows_metrics[key] = row
                best_key = max(
                    target_rows_metrics,
                    key=lambda key: target_rows_metrics[key]["proxy_score"],
                )
                per_target.append(
                    {
                        "target": target,
                        "source": source_name,
                        "best_proxy_config": best_key,
                    }
                )
                print(
                    json.dumps(
                        {
                            "completed": len(per_target),
                            "total": len(eligible),
                            "target": target,
                        }
                    ),
                    flush=True,
                )
        finally:
            adata.file.close()

    aggregated = {key: aggregate(rows) for key, rows in results.items()}
    ranking = sorted(
        aggregated,
        key=lambda key: aggregated[key]["proxy_score"],
        reverse=True,
    )
    payload = {
        "targets": len(eligible),
        "configurations": len(configurations),
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
