"""Evaluate reciprocal MagWorld checkpoints and their zero-shot ensemble."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_magworld_h1_v3 import calibrate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signatures", required=True)
    parser.add_argument("--official-perts", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--magworld-src", required=True)
    parser.add_argument("--out")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def predict_checkpoint(
    path: str,
    model_class,
    controls: np.ndarray,
    target_gene_indices: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, object], np.ndarray]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = model_class(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    with torch.no_grad():
        prediction = model.predict_delta(
            torch.from_numpy(controls).to(device),
            torch.from_numpy(target_gene_indices).to(device),
        ).cpu().numpy()
    return checkpoint, prediction


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    module_path = str(Path(args.magworld_src).resolve())
    if module_path not in sys.path:
        sys.path.insert(0, module_path)
    model_class = importlib.import_module("model_world_h1_v3").WorldModelH1V3

    dataset = np.load(args.signatures, allow_pickle=False)
    genes = dataset["genes"].astype(str).tolist()
    targets = dataset["targets"].astype(str)
    controls = dataset["controls"].astype(np.float32)
    effects = dataset["effects"].astype(np.float32)
    official = set(pd.read_csv(args.official_perts)["target_gene"].astype(str))
    validation_indices = np.asarray(
        [index for index, target in enumerate(targets) if target in official],
        dtype=np.int64,
    )
    gene_lookup = {gene: index for index, gene in enumerate(genes)}
    validation_gene_indices = np.asarray(
        [gene_lookup[target] for target in targets[validation_indices]], dtype=np.int64
    )

    predictions: list[np.ndarray] = []
    results: list[dict[str, object]] = []
    for path in args.checkpoints:
        checkpoint, prediction = predict_checkpoint(
            path,
            model_class,
            controls[validation_indices],
            validation_gene_indices,
            device,
        )
        if checkpoint["genes"] != genes:
            raise ValueError(f"checkpoint gene order differs: {path}")
        calibration = calibrate(
            prediction, effects[validation_indices], validation_gene_indices
        )
        sparse_calibration = calibrate(
            prediction,
            effects[validation_indices],
            validation_gene_indices,
            top_k_values=(100, 200, 500),
        )
        predictions.append(prediction)
        results.append(
            {
                "checkpoint": path,
                "best_epoch": checkpoint.get("best_epoch"),
                "calibration": calibration,
                "sparse_calibration": sparse_calibration,
            }
        )

    ensembles: list[dict[str, object]] = []
    if len(predictions) == 2:
        for first_weight in np.linspace(0.0, 1.0, 11):
            prediction = (
                first_weight * predictions[0]
                + (1.0 - first_weight) * predictions[1]
            )
            calibration = calibrate(
                prediction, effects[validation_indices], validation_gene_indices
            )
            sparse_calibration = calibrate(
                prediction,
                effects[validation_indices],
                validation_gene_indices,
                top_k_values=(100, 200, 500),
            )
            ensembles.append(
                {
                    "weights": [float(first_weight), float(1.0 - first_weight)],
                    "calibration": calibration,
                    "sparse_calibration": sparse_calibration,
                }
            )

    best_candidates = [
        {"kind": "single", "weights": [float(i == index) for i in range(len(results))],
         **result}
        for index, result in enumerate(results)
    ]
    best_candidates.extend(
        {"kind": "ensemble", **result} for result in ensembles
    )
    best = max(
        best_candidates,
        key=lambda value: float(value["calibration"]["objective"]),
    )
    best_sparse = max(
        best_candidates,
        key=lambda value: float(value["sparse_calibration"]["objective"]),
    )
    report = {
        "validation_targets": targets[validation_indices].tolist(),
        "models": results,
        "ensembles": ensembles,
        "best": best,
        "best_sparse": best_sparse,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered, flush=True)
    if args.out:
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
