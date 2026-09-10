# Probabilistic Tsunami Surrogate

A probabilistic neural surrogate for tsunami simulations in the Makran
subduction zone. Given an initial surface-elevation field, bathymetry and
scenario metadata, the model estimates maximum wave height and first-wave
arrival time at a fixed network of 317 coastal stations.

The accompanying [report](report.pdf) has been edited for presentation and
clarity; the reported results are unchanged. The original simulation data and
trained checkpoint cannot be redistributed, so this repository documents the
research pipeline rather than providing a self-contained reproducible release.
Tests use small synthetic fixtures.

## Repository structure

```text
src/probabilistic_tsunami_surrogate/
  config.py          paths and run settings
  data/
    cleaning.py      station parsing and target construction
    dataset.py       PyTorch dataset
    preprocessing.py scenario discovery and transformations
    splits.py        train/test and cross-validation splits
    validation.py    data checks and target summaries
  evaluation.py      checkpoint prediction and evaluation
  losses.py          multitask Student-t likelihood
  metrics.py         deterministic and probabilistic metrics
  model.py           station-aware ResNet-18
  plotting.py        dataset and evaluation figures
  training.py        training loop
scripts/             preprocessing, training and evaluation entry points
tests/               unit tests
```

## Environment

Python 3.10 or newer is required.

This setup is provided for running the synthetic tests and for researchers who
have access to the original data. Create the environment on the workstation or
hosted runtime where the code will be executed. A GPU environment must have a
CUDA-compatible build of PyTorch and torchvision; hosted GPU runtimes such as
Google Colab commonly provide these already. The project and all of its
optional dependencies can then be installed with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[model,evaluation,report,test]"
pytest -q
```

In Colab, the equivalent package installation can be run with `%pip` from the
repository directory. These commands install the software only; they do not
provide the simulation data or a trained checkpoint.

## Compute

For computational provenance, the full-resolution model was originally trained
using a GPU in Google Colab. For a rerun with the source data, a CUDA-capable
GPU is strongly recommended because CPU execution is prohibitively slow at the
working grid size. Training and evaluation select CUDA automatically when
`torch.cuda.is_available()` is true, while retaining a CPU fallback for small
checks. No code changes are needed when moving to a CUDA-enabled machine; only
the appropriate PyTorch installation and the project dependencies are required
there.

## Data

The simulation dataset belongs to the originating research group and is not
available for redistribution. It is not included here and is not intended for
public release. The layout below documents the structure expected by the code;
it is not a description of a downloadable dataset:

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
  9.0/
```

Paths and run settings are defined in `RunConfig` in
`src/probabilistic_tsunami_surrogate/config.py`.

## Pipeline reference

The following commands document the research workflow. They require access to
the original data in the expected directory structure and are not a publicly
reproducible example.

Prepare and validate the data:

```bash
python scripts/convert_text_arrays.py       # if the arrays are stored as text
python scripts/build_scenario_targets.py    # if targets must be rebuilt
python scripts/build_entries.py
python scripts/validate_data.py
python scripts/build_splits.py
```

Train and evaluate the model:

```bash
python scripts/train_model.py
python scripts/evaluate_model.py
python scripts/build_report_assets.py
```

The trained checkpoint is also not distributed with the repository.
