# MagWorld Latest

MagWorld is a zero-shot perturbation-response model for the 2026 Virtual Cell
Challenge. It maps pretrained gene features into directed reciprocal fields,
predicts a log-fold-change signature for each CRISPRi target, and decodes that
signature into raw single-cell counts with a Gamma-Poisson model.

This repository contains the reproducible model and inference code. Challenge
data, checkpoints, generated `.h5ad` files, and `.vcc` archives are deliberately
excluded because they are large and may be access-controlled.

## Current status

The published v4 submission `magworld_h1_v4_directed_cv5_ensemble` used three
full-data seeds with equal weights, `top_k=500`, `downstream_scale=2.0`, and
`prior_strength=2.0`. At the time recorded, it ranked 183/272 with an overall
scaled score of -0.0448.

The v5 code retains per-seed predictions and adds experimental inference
switches that can:

- remove a configurable fraction of the response shared by all 300 targets;
- shrink genes whose magnitude or direction is unstable across seeds;
- suppress impossible down-regulation for genes absent from a context;
- rank decoder settings with a balanced proxy for the six VCC 2026 metrics.

None of these switches improved the five-fold aggregate consistently. The
recommended configuration therefore preserves the published v4 behavior:
`panel_centering=0`, `min_sign_agreement=0`, `uncertainty_penalty=0`, and
`expression_gate_scale=0`. The switches remain available for future experiments,
but are not claimed as score improvements.

## Repository layout

```text
src/
  build_h1_signature_dataset.py    batch-matched H1 signatures
  build_hybrid_gene_embeddings.py  scGPT + control co-expression features
  model_world_h1_v4.py             directed reciprocal world model
  train_magworld_h1_v4.py          zero-shot training and calibration
  evaluate_decoder_h1_v4.py        distributional v4 decoder evaluation
  evaluate_decoder_h1_v5.py        v5 panel-centering/consensus grid
  distributional_decoder.py        strict distribution diagnostics and optional cell heterogeneity
  predict_magworld_vcc2026_v4.py   published-v4 inference
  predict_magworld_vcc2026_v5.py   robust-consensus v5 inference
tests/                              focused model and decoder tests
```

## Environment

Python 3.12 and a CUDA-capable PyTorch installation are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src pytest -q
```

## Evaluate v5

The strict test-lock below evaluates only targets shared by the official VCC
panel and the held-out H1 test split.

```bash
PYTHONPATH=src python src/evaluate_decoder_h1_v5.py \
  --checkpoints checkpoints/magworld_h1_v4_full_seed113.pt \
                checkpoints/magworld_h1_v4_full_seed227.pt \
                checkpoints/magworld_h1_v4_full_seed337.pt \
  --ensemble-weights 1 1 1 \
  --h1 data/h1_2025/adata_Test.h5ad \
  --signatures data/h1_2025/h1_test_signatures_v4.npz \
  --genes vcc_data/gene_names.csv \
  --perts vcc_data/pert_counts.csv \
  --magworld-src src \
  --out results/decoder_v5_testlock.json \
  --top-k 350 500 \
  --downstream-scale 1.5 2.0 \
  --panel-centering 0 0.5 1.0 \
  --min-sign-agreement 0 \
  --uncertainty-penalty 0 \
  --expression-gate-scale 0 \
  --device cuda:0
```

## Generate a prediction

```bash
PYTHONPATH=src python src/predict_magworld_vcc2026_v5.py \
  --checkpoints checkpoints/magworld_h1_v4_full_seed113.pt \
                checkpoints/magworld_h1_v4_full_seed227.pt \
                checkpoints/magworld_h1_v4_full_seed337.pt \
  --ensemble-weights 1 1 1 \
  --controls-dir vcc_data \
  --genes vcc_data/gene_names.csv \
  --perts vcc_data/pert_counts.csv \
  --magworld-src src \
  --top-k 500 \
  --downstream-scale 2.0 \
  --panel-centering 0 \
  --expression-gate-scale 0 \
  --prior-strength 2.0 \
  --out predictions/magworld_h1_v5_robust_consensus.h5ad \
  --device cuda:0
```

Package the generated raw-count file with the official VCC CLI after validating
its contexts, targets, gene order, and cell counts.

## Validation snapshot

## v6 distribution experiments

The v6 evaluation adds a blind all-gene mode (`--strict-all-genes`) and reports
latent PCA-FID, library-size distance, and detected-gene distance. Every decoder
configuration uses the same random seed per target so comparisons do not include
configuration-specific Monte Carlo noise.

On the eight-target blind test-lock, matching decoded library sizes to controls
was not useful:

| Library match strength | Proxy | Latent FID |
|---:|---:|---:|
| 0 | 0.32275 | 10.733 |
| 0.5 | 0.32217 | 10.743 |
| 1.0 | 0.32221 | 10.750 |

Cell-level response heterogeneity (`--response-sigma`) had a small test-lock
gain at `0.30`, but the five-fold, 25-target aggregate was effectively tied:
`0.25055` for sigma 0, `0.25061` for 0.15, and `0.25056` for 0.30. It remains
experimental and is disabled by default.

Lowering the decoder prior from 2.0 to 0.5 improved the eight-target blind
proxy from `0.32276` to `0.32472` and reduced latent FID from `10.733` to
`10.592`, but this has not yet completed an independent five-fold confirmation.
The published/default configuration therefore remains `prior_strength=2.0`.

The five-fold aggregate decoder proxy favored no panel centering:

| Downstream scale | Panel centering | Proxy score |
|---:|---:|---:|
| 2.0 | 0 | 0.28441 |
| 1.5 | 0 | 0.27830 |
| 2.0 | 0.5 | 0.25647 |
| 1.5 | 0.5 | 0.24840 |

No centering with scale 2.0 won four of five folds; scale 1.5 won the remaining
fold. Consensus shrinkage and the expression gate also lacked stable gains.

On the eight-target independent H1 test-lock, the baseline and one experimental
centering candidate produced:

| Setting | Fidelity | Signed reach | Jaccard | LFC NMAE | Cosine |
|---|---:|---:|---:|---:|---:|
| v4 / center 0 | 0.942 | 0.295 | 0.163 | 0.864 | 0.424 |
| experimental / center 0.5 | 0.961 | 0.260 | 0.144 | 0.839 | 0.437 |

Although centering improved some test-lock metrics, it reduced signed reach and
Jaccard and performed worse on the broader five-fold proxy. It is therefore off
by default.

Two fold-0 training ablations for the shared-response bias were also rejected.
Removing the bias reached a best objective of 0.18012, and stronger bias
regularization reached 0.17775, both below the existing fold-0 baseline of
0.23066. No new online VCC submission was produced from these regressions.
