"""Train directed MagWorld with DE-yield-aligned objectives on H1 targets."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signatures", required=True)
    parser.add_argument("--official-perts", required=True)
    parser.add_argument("--gene-embeddings", required=True)
    parser.add_argument("--magworld-src", required=True)
    parser.add_argument("--model-module", default="model_world_h1_v4")
    parser.add_argument("--model-class", default="WorldModelH1V4")
    parser.add_argument("--sparse-top-k", type=int, default=2048)
    parser.add_argument("--cheb-order", type=int, default=3)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--init-checkpoint",
        help="Warm-start from a compatible MagWorld checkpoint before training.",
    )
    parser.add_argument(
        "--init-mode",
        choices=("full", "field-only"),
        default="full",
        help="Load the whole model or only transferable perturbation-field weights.",
    )
    parser.add_argument(
        "--holdout-mode", choices=("official-overlap", "random", "none"),
        default="official-overlap",
    )
    parser.add_argument("--random-holdout", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--lambda-global", type=float, default=0.2)
    parser.add_argument("--lambda-de", type=float, default=1.0)
    parser.add_argument("--lambda-cosine", type=float, default=0.5)
    parser.add_argument("--lambda-direction", type=float, default=0.15)
    parser.add_argument("--lambda-rank", type=float, default=0.1)
    parser.add_argument("--lambda-pds", type=float, default=0.05)
    parser.add_argument("--lambda-yield", type=float, default=0.15)
    parser.add_argument("--lambda-shared-bias", type=float, default=0.05)
    parser.add_argument("--yield-threshold", type=float, default=0.05)
    parser.add_argument("--yield-temperature", type=float, default=0.02)
    parser.add_argument("--direction-temperature", type=float, default=0.05)
    parser.add_argument("--rank-temperature", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=0.15)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument(
        "--magnetic-mode", choices=("average", "directed"), default="directed"
    )
    parser.add_argument(
        "--shared-bias-mode", choices=("free", "gated", "none"), default="gated"
    )
    parser.add_argument("--shared-bias-initial-scale", type=float, default=0.1)
    parser.add_argument(
        "--normalize-context", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def keep_top_k(
    values: np.ndarray,
    k: int,
    target_gene_indices: np.ndarray | None = None,
) -> np.ndarray:
    if k < 0 or k >= values.shape[1] - 1:
        return values.copy()
    magnitude = np.abs(values).copy()
    if target_gene_indices is not None:
        magnitude[np.arange(len(values)), target_gene_indices] = -np.inf
    indices = np.argpartition(magnitude, -k, axis=1)[:, -k:]
    result = np.zeros_like(values)
    np.put_along_axis(result, indices, np.take_along_axis(values, indices, axis=1), axis=1)
    if target_gene_indices is not None:
        rows = np.arange(len(values))
        result[rows, target_gene_indices] = values[rows, target_gene_indices]
    return result


def signature_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    target_gene_indices: np.ndarray,
    evaluation_k: int = 100,
) -> dict[str, float]:
    eps = 1e-8
    cosine = np.sum(prediction * truth, axis=1) / np.maximum(
        np.linalg.norm(prediction, axis=1) * np.linalg.norm(truth, axis=1), eps
    )
    centered_prediction = prediction - prediction.mean(1, keepdims=True)
    centered_truth = truth - truth.mean(1, keepdims=True)
    pearson = np.sum(centered_prediction * centered_truth, axis=1) / np.maximum(
        np.linalg.norm(centered_prediction, axis=1)
        * np.linalg.norm(centered_truth, axis=1),
        eps,
    )
    nmae = np.mean(np.abs(prediction - truth), axis=1) / np.maximum(
        np.mean(np.abs(truth), axis=1), eps
    )
    pred_top = np.argpartition(np.abs(prediction), -evaluation_k, axis=1)[:, -evaluation_k:]
    true_top = np.argpartition(np.abs(truth), -evaluation_k, axis=1)[:, -evaluation_k:]
    jaccard: list[float] = []
    reach: list[float] = []
    fidelity: list[float] = []
    signed_reach: list[float] = []
    yield_similarity: list[float] = []
    for row in range(len(prediction)):
        pred_set = set(pred_top[row].tolist())
        true_set = set(true_top[row].tolist())
        overlap = np.asarray(sorted(pred_set & true_set), dtype=np.int64)
        union = pred_set | true_set
        jaccard.append(len(overlap) / max(1, len(union)))
        reach.append(len(overlap) / evaluation_k)
        fidelity.append(
            float(
                np.mean(
                    np.sign(prediction[row, overlap])
                    == np.sign(truth[row, overlap])
                )
            )
            if len(overlap)
            else 0.0
        )
        sign_matches = (
            int(
                np.sum(
                    np.sign(prediction[row, overlap])
                    == np.sign(truth[row, overlap])
                )
            )
            if len(overlap)
            else 0
        )
        signed_reach.append(sign_matches / evaluation_k)
        predicted_material = int(np.sum(np.abs(prediction[row]) >= 0.05))
        truth_material = int(np.sum(np.abs(truth[row]) >= 0.05))
        yield_similarity.append(
            min(predicted_material, truth_material)
            / max(1, predicted_material, truth_material)
        )
    rows = np.arange(len(prediction))
    self_truth = truth[rows, target_gene_indices]
    self_prediction = prediction[rows, target_gene_indices]
    nonzero_self = np.abs(self_truth) > 1e-6
    self_direction = float(
        np.mean(np.sign(self_prediction[nonzero_self]) == np.sign(self_truth[nonzero_self]))
    ) if np.any(nonzero_self) else 0.0
    return {
        "cosine": float(np.mean(cosine)),
        "pearson": float(np.mean(pearson)),
        "mae": float(np.mean(np.abs(prediction - truth))),
        "nmae": float(np.mean(nmae)),
        "direction_fidelity": float(np.mean(fidelity)),
        "signed_reach": float(np.mean(signed_reach)),
        "reach": float(np.mean(reach)),
        "jaccard": float(np.mean(jaccard)),
        "yield_similarity": float(np.mean(yield_similarity)),
        "self_direction": self_direction,
        "effect_l1": float(np.mean(np.abs(prediction))),
    }


def metric_objective(metrics: dict[str, float]) -> float:
    return (
        metrics["cosine"]
        + 0.5 * metrics["signed_reach"]
        + 0.25 * metrics["jaccard"]
        + 0.1 * metrics["yield_similarity"]
        - 0.15 * metrics["nmae"]
    )


def calibrate(
    prediction: np.ndarray,
    truth: np.ndarray,
    target_gene_indices: np.ndarray,
    top_k_values: tuple[int, ...] = (50, 100, 200, 500, -1),
) -> dict[str, object]:
    rows = np.arange(len(prediction))
    predicted_self = prediction[rows, target_gene_indices]
    true_self = truth[rows, target_gene_indices]
    self_denominator = float(np.dot(predicted_self, predicted_self))
    self_scale = float(
        np.clip(
            np.dot(predicted_self, true_self) / max(self_denominator, 1e-8),
            0.25,
            2.0,
        )
    )
    candidates: list[dict[str, object]] = []
    for top_k in top_k_values:
        sparse_prediction = keep_top_k(prediction, top_k, target_gene_indices)
        for downstream_scale in (0.1, 0.25, 0.5, 0.75, 1.0, 1.25):
            calibrated_prediction = downstream_scale * sparse_prediction
            calibrated_prediction[rows, target_gene_indices] = (
                self_scale * predicted_self
            )
            metrics = signature_metrics(
                calibrated_prediction, truth, target_gene_indices
            )
            candidates.append(
                {
                    "top_k": top_k,
                    "downstream_scale": downstream_scale,
                    "self_scale": self_scale,
                    "objective": metric_objective(metrics),
                    "metrics": metrics,
                }
            )
    return max(candidates, key=lambda value: float(value["objective"]))


def ridge_baseline(
    features: np.ndarray,
    effects: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    target_gene_indices: np.ndarray,
) -> dict[str, object]:
    train_x = features[train_indices]
    validation_x = features[validation_indices]
    train_y = effects[train_indices]
    x_mean = train_x.mean(0, keepdims=True)
    y_mean = train_y.mean(0, keepdims=True)
    centered_x = train_x - x_mean
    centered_y = train_y - y_mean
    best: dict[str, object] | None = None
    identity = np.eye(centered_x.shape[1], dtype=np.float32)
    covariance = centered_x.T @ centered_x
    cross_covariance = centered_x.T @ centered_y
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
        weights = np.linalg.solve(
            covariance + alpha * identity,
            cross_covariance,
        )
        prediction = (validation_x - x_mean) @ weights + y_mean
        calibrated = calibrate(
            prediction, effects[validation_indices], target_gene_indices
        )
        result = {"alpha": alpha, **calibrated}
        if best is None or float(result["objective"]) > float(best["objective"]):
            best = result
    assert best is not None
    return best


def nearest_neighbor_baseline(
    features: np.ndarray,
    effects: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    target_gene_indices: np.ndarray,
) -> dict[str, object]:
    train_x = features[train_indices]
    validation_x = features[validation_indices]
    similarity = validation_x @ train_x.T
    best: dict[str, object] | None = None
    for neighbors in (3, 5, 10, 20):
        indices = np.argpartition(similarity, -neighbors, axis=1)[:, -neighbors:]
        selected = np.take_along_axis(similarity, indices, axis=1)
        weights = np.exp((selected - selected.max(1, keepdims=True)) / 0.1)
        weights /= weights.sum(1, keepdims=True)
        prediction = np.sum(
            effects[train_indices][indices] * weights[:, :, None], axis=1
        )
        calibrated = calibrate(
            prediction, effects[validation_indices], target_gene_indices
        )
        result = {"neighbors": neighbors, **calibrated}
        if best is None or float(result["objective"]) > float(best["objective"]):
            best = result
    assert best is not None
    return best


def pds_loss(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    labels: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    pred_direction = F.normalize(prediction, dim=-1)
    truth_direction = F.normalize(truth, dim=-1)
    similarity = pred_direction @ truth_direction.T / tau
    positives = labels[:, None] == labels[None, :]

    def one_direction(logits: torch.Tensor) -> torch.Tensor:
        log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        return -(
            (log_probability * positives).sum(1)
            / positives.sum(1).clamp_min(1)
        ).mean()

    return 0.5 * (one_direction(similarity) + one_direction(similarity.T))


def de_yield_loss(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    threshold: float,
    temperature: float,
) -> torch.Tensor:
    """Match the number of material effects with a differentiable count."""
    if threshold <= 0.0 or temperature <= 0.0:
        raise ValueError("yield threshold and temperature must be positive")
    predicted_count = torch.sigmoid(
        (prediction.abs() - threshold) / temperature
    ).sum(1)
    truth_count = torch.sigmoid((truth.abs() - threshold) / temperature).sum(1)
    return F.smooth_l1_loss(
        torch.log1p(predicted_count), torch.log1p(truth_count), reduction="none"
    )


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )

    dataset = np.load(args.signatures, allow_pickle=False)
    genes = dataset["genes"].astype(str).tolist()
    targets = dataset["targets"].astype(str)
    controls = dataset["controls"].astype(np.float32)
    effects = dataset["effects"].astype(np.float32)
    replicate_controls = dataset["replicate_controls"].astype(np.float32)
    replicate_effects = dataset["replicate_effects"].astype(np.float32)
    replicate_targets = dataset["replicate_targets"].astype(np.int64)
    replicate_cells = dataset["replicate_cells"].astype(np.float32)
    features = np.load(args.gene_embeddings).astype(np.float32)
    if features.shape[0] != len(genes):
        raise ValueError("gene embeddings and signature genes differ")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.maximum(norms, 1e-8)
    gene_lookup = {gene: index for index, gene in enumerate(genes)}
    target_gene_indices = np.asarray([gene_lookup[target] for target in targets])
    official = set(pd.read_csv(args.official_perts)["target_gene"].astype(str))

    if args.holdout_mode == "official-overlap":
        validation_indices = np.asarray(
            [index for index, target in enumerate(targets) if target in official],
            dtype=np.int64,
        )
    elif args.holdout_mode == "random":
        validation_indices = np.sort(
            rng.choice(len(targets), min(args.random_holdout, len(targets) // 3), replace=False)
        )
    else:
        validation_indices = np.asarray([], dtype=np.int64)
    validation_set = set(validation_indices.tolist())
    train_indices = np.asarray(
        [index for index in range(len(targets)) if index not in validation_set],
        dtype=np.int64,
    )
    if len(validation_indices) == 0:
        validation_indices = train_indices
    train_set = set(train_indices.tolist())
    replicate_train = np.asarray(
        [index for index, target in enumerate(replicate_targets) if int(target) in train_set],
        dtype=np.int64,
    )

    validation_target_genes = target_gene_indices[validation_indices]
    baseline = {
        "zero": {
            "top_k": -1,
            "scale": 0.0,
            "metrics": signature_metrics(
                np.zeros_like(effects[validation_indices]),
                effects[validation_indices],
                validation_target_genes,
            ),
        },
        "mean": calibrate(
            np.repeat(effects[train_indices].mean(0, keepdims=True), len(validation_indices), axis=0),
            effects[validation_indices],
            validation_target_genes,
        ),
        "ridge": ridge_baseline(
            features[target_gene_indices],
            effects,
            train_indices,
            validation_indices,
            validation_target_genes,
        ),
        "nearest_neighbor": nearest_neighbor_baseline(
            features[target_gene_indices],
            effects,
            train_indices,
            validation_indices,
            validation_target_genes,
        ),
    }
    baseline["zero"]["objective"] = metric_objective(baseline["zero"]["metrics"])
    print(json.dumps({"baselines": baseline}), flush=True)

    module_path = str(Path(args.magworld_src).resolve())
    if module_path not in sys.path:
        sys.path.insert(0, module_path)
    model_module = importlib.import_module(args.model_module)
    model_class = getattr(model_module, args.model_class)
    model_config = {
        "n_genes": len(genes),
        "d_model": features.shape[1],
        "d_z": args.latent_dim,
        "d_hidden": args.hidden_dim,
        "directed_magnetic": args.magnetic_mode == "directed",
        "shared_bias_mode": args.shared_bias_mode,
        "shared_bias_initial_scale": args.shared_bias_initial_scale,
        "normalize_context": args.normalize_context,
    }
    if "sparse_top_k" in inspect.signature(model_class.__init__).parameters:
        model_config["sparse_top_k"] = args.sparse_top_k
    if "cheb_order" in inspect.signature(model_class.__init__).parameters:
        model_config["cheb_order"] = args.cheb_order
    model = model_class(**model_config).to(device)
    model.initialize_gene_embeddings(torch.from_numpy(features).to(device))
    initialization: dict[str, object] | None = None
    if args.init_checkpoint:
        init_path = Path(args.init_checkpoint)
        source = torch.load(init_path, map_location=device, weights_only=False)
        source_genes = np.asarray(source.get("genes", []), dtype=str)
        if not np.array_equal(source_genes, genes):
            raise ValueError(
                "initialization checkpoint gene order differs from the signature dataset"
            )
        source_config = source.get("model_config", {})
        compared_keys = (
            model_config.keys()
            if args.init_mode == "full"
            else ("n_genes", "d_model", "d_z", "d_hidden", "directed_magnetic")
        )
        incompatible = {
            key: (source_config.get(key), value)
            for key in compared_keys
            if (value := model_config.get(key)) is not None
            if source_config.get(key) != value
        }
        if incompatible:
            raise ValueError(
                f"initialization checkpoint model config is incompatible: {incompatible}"
            )
        if args.init_mode == "full":
            transferred_state = source["model_state"]
            model.load_state_dict(transferred_state, strict=True)
        else:
            transferable_prefixes = ("magnet.", "source_code.", "receiver_code.")
            transferable_names = {"receiver_residual", "magnetic_mix"}
            target_state = model.state_dict()
            transferred_state = {
                key: value
                for key, value in source["model_state"].items()
                if (key.startswith(transferable_prefixes) or key in transferable_names)
                and key in target_state
                and value.shape == target_state[key].shape
            }
            if not transferred_state:
                raise ValueError("initialization checkpoint has no transferable field weights")
            model.load_state_dict(transferred_state, strict=False)
        initialization = {
            "checkpoint": str(init_path),
            "mode": args.init_mode,
            "transferred_parameters": sorted(transferred_state),
            "best_epoch": source.get("best_epoch"),
            "best_validation": source.get("best_validation"),
            "features_path": source.get("features_path"),
        }
        print(json.dumps({"initialization": initialization}), flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    canonical_abs = np.abs(effects)
    canonical_top = np.argpartition(canonical_abs, -args.top_k, axis=1)[:, -args.top_k:]
    de_masks = np.zeros_like(effects, dtype=bool)
    np.put_along_axis(de_masks, canonical_top, True, axis=1)
    sample_weight = np.sqrt(replicate_cells[replicate_train].clip(min=1.0))
    sample_weight /= sample_weight.mean()

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    best_objective = -math.inf
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(replicate_train))
        totals = {"loss": 0.0, "global": 0.0, "de": 0.0, "cosine": 0.0,
                  "direction": 0.0, "rank": 0.0, "pds": 0.0, "yield": 0.0,
                  "shared_bias": 0.0}
        batches = 0
        for start in range(0, len(order), args.batch_size):
            positions = order[start : start + args.batch_size]
            sample_indices = replicate_train[positions]
            target_indices = replicate_targets[sample_indices]
            xb = torch.from_numpy(replicate_controls[sample_indices]).to(device)
            yb = torch.from_numpy(replicate_effects[sample_indices]).to(device)
            pert = torch.from_numpy(target_gene_indices[target_indices]).to(device)
            mask = torch.from_numpy(de_masks[target_indices]).to(device)
            weights = torch.from_numpy(sample_weight[positions]).to(device)

            prediction = model.predict_delta(xb, pert)
            absolute_error = (prediction - yb).abs()
            global_per_sample = absolute_error.mean(1)
            de_per_sample = (absolute_error * mask).sum(1) / mask.sum(1).clamp_min(1)
            cosine_per_sample = 1.0 - F.cosine_similarity(prediction, yb, dim=1)
            truth_sign = torch.sign(yb)
            magnitude = yb.abs()
            direction_penalty = F.softplus(
                -truth_sign * prediction / args.direction_temperature
            )
            direction_per_sample = (
                direction_penalty * magnitude * mask
            ).sum(1) / (magnitude * mask).sum(1).clamp_min(1e-6)
            rank_target = magnitude * mask
            rank_target = rank_target / rank_target.sum(1, keepdim=True).clamp_min(1e-6)
            rank_per_sample = -(
                rank_target
                * F.log_softmax(prediction.abs() / args.rank_temperature, dim=1)
            ).sum(1)
            discrimination = pds_loss(
                prediction, yb, torch.from_numpy(target_indices).to(device), args.tau
            )
            yield_per_sample = de_yield_loss(
                prediction,
                yb,
                args.yield_threshold,
                args.yield_temperature,
            )
            global_loss = (global_per_sample * weights).mean()
            de_loss = (de_per_sample * weights).mean()
            cosine_loss = (cosine_per_sample * weights).mean()
            direction_loss = (direction_per_sample * weights).mean()
            rank_loss = (rank_per_sample * weights).mean()
            yield_loss = (yield_per_sample * weights).mean()
            shared_bias_loss = (
                model.effective_shared_bias_scale() * model.gene_bias.abs().mean()
            )
            loss = (
                args.lambda_global * global_loss
                + args.lambda_de * de_loss
                + args.lambda_cosine * cosine_loss
                + args.lambda_direction * direction_loss
                + args.lambda_rank * rank_loss
                + args.lambda_pds * discrimination
                + args.lambda_yield * yield_loss
                + args.lambda_shared_bias * shared_bias_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            values = {
                "loss": loss,
                "global": global_loss,
                "de": de_loss,
                "cosine": cosine_loss,
                "direction": direction_loss,
                "rank": rank_loss,
                "pds": discrimination,
                "yield": yield_loss,
                "shared_bias": shared_bias_loss,
            }
            for key, value in values.items():
                totals[key] += float(value.detach())
            batches += 1

        model.eval()
        with torch.no_grad():
            validation_prediction = model.predict_delta(
                torch.from_numpy(controls[validation_indices]).to(device),
                torch.from_numpy(validation_target_genes).to(device),
            ).cpu().numpy()
        calibration = calibrate(
            validation_prediction,
            effects[validation_indices],
            validation_target_genes,
        )
        summary: dict[str, object] = {
            "epoch": epoch,
            "training": {key: value / max(1, batches) for key, value in totals.items()},
            "validation": calibration,
            "magnetic": model.magnetic_diagnostics(
                torch.from_numpy(validation_target_genes[: min(16, len(validation_target_genes))]).to(device)
            ),
        }
        history.append(summary)
        print(json.dumps(summary), flush=True)
        objective = float(calibration["objective"])
        save_checkpoint = (
            objective > best_objective or args.holdout_mode == "none"
        )
        if save_checkpoint:
            best_objective = objective
            best_epoch = epoch
            stale_epochs = 0
            checkpoint = {
                "model_state": model.state_dict(),
                "model_config": model_config,
                "genes": genes,
                "features_path": args.gene_embeddings,
                "signature_targets": targets.tolist(),
                "signature_effects": effects,
                "signature_controls": controls,
                "train_targets": targets[train_indices].tolist(),
                "validation_targets": targets[validation_indices].tolist(),
                "baseline": baseline,
                "best_epoch": best_epoch,
                "best_validation": calibration,
                "checkpoint_selection": (
                    "last_epoch" if args.holdout_mode == "none" else "validation_objective"
                ),
                "initialization": initialization,
                "history": history,
                "training_args": vars(args),
            }
            temporary = output.with_suffix(output.suffix + ".tmp")
            torch.save(checkpoint, temporary)
            os.replace(temporary, output)
            output.with_suffix(".json").write_text(
                json.dumps(
                    {key: value for key, value in checkpoint.items()
                     if key not in {"model_state", "signature_effects", "signature_controls"}},
                    indent=2,
                ) + "\n",
                encoding="utf-8",
            )
        else:
            stale_epochs += 1
        if args.holdout_mode != "none" and stale_epochs >= args.patience:
            print(
                json.dumps({"early_stop": epoch, "best_epoch": best_epoch,
                            "best_objective": best_objective}),
                flush=True,
            )
            break
    print(
        json.dumps({"checkpoint": str(output), "best_epoch": best_epoch,
                    "best_objective": best_objective}),
        flush=True,
    )


if __name__ == "__main__":
    main()
