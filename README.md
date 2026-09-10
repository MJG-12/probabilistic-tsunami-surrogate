# Probabilistic Tsunami Surrogate

A probabilistic neural surrogate for tsunami simulations in the Makran
subduction zone. Given an initial surface-elevation field, bathymetry and
earthquake slip type, the model predicts maximum wave height and first-wave
arrival time at 317 coastal stations.

This repository contains the research code accompanying the [report](report.pdf).
The code was extracted from research notebooks and reorganized into modules
and scripts for readability. The original research was carried out in stages
across those notebooks.

The original simulation data and trained checkpoint are not available for
redistribution.

## Model

A modified ResNet-18 combines global image features with local features sampled
at station locations, station coordinates and slip embeddings. A shared decoder
predicts Student-t distribution parameters for both tasks, providing point
predictions and prediction intervals. The report describes the methodology,
evaluation and limitations.

## Code structure

```text
src/probabilistic_tsunami_surrogate/
  config.py          paths and run settings
  data/              target construction, filtering, preprocessing and splits
  model.py           station-aware ResNet-18
  losses.py          multitask Student-t likelihood
  training.py        training and validation
  evaluation.py      checkpoint loading and prediction
  metrics.py         deterministic and probabilistic evaluation
  plotting.py        data and evaluation figures
scripts/             entry points for individual research stages
tests/               checks using synthetic data
```

| Script | Purpose |
| --- | --- |
| `convert_text_arrays.py` | Convert simulation text arrays to NumPy format |
| `build_scenario_targets.py` | Extract wave heights and arrival times from station outputs |
| `build_entries.py` | Filter scenarios and remove duplicates |
| `validate_data.py` | Check data quality and summarize target distributions |
| `build_splits.py` | Create magnitude-stratified train/test and validation splits |
| `train_model.py` | Train the surrogate |
| `evaluate_model.py` | Evaluate a checkpoint on the test set |
| `build_report_assets.py` | Generate evaluation figures and summaries |
| `evaluate_reference.py` | Calculate reference Student-t CRPS scores |

## Setup

Requires Python 3.10 or newer. To install the project and its optional
dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[model,evaluation,report,test]"
pytest -q
```

The tests use synthetic data. Training and evaluation require the original
simulation data. The model was trained on a GPU in Google Colab; full-resolution
training is intended for a CUDA-capable GPU. CPU execution is supported for
small checks.

## Data layout

The code expects the following directory structure:

```text
data/raw/
  additional_files/
    bathy.npy
    stationsall.npy
  8.9/
    M1/
      external_files/eta.npy
      hmax.npy
      arrival_times.npy
      outputs/sta_0001
      ...
    M2/
  9/
```

Paths and run settings are defined in
[config.py](src/probabilistic_tsunami_surrogate/config.py).
Run individual stages from the repository root, for example:

```bash
python scripts/train_model.py
python scripts/evaluate_model.py
```

Reference CRPS evaluation also requires the saved target statistics in
`REFERENCE_STATS_DIR` (default `data/stats`): `hmax_mean_asinh.npy`,
`hmax_std_asinh.npy`, `arrival_time_mean_log.npy` and
`arrival_time_std_log.npy`.
