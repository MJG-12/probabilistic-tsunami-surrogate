"""Checkpoint evaluation for the station-aware surrogate."""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from probabilistic_tsunami_surrogate.data.dataset import (
    MakranDataset,
    station_pixel_coordinates,
)
from probabilistic_tsunami_surrogate.metrics import summarize_predictions
from probabilistic_tsunami_surrogate.model import StationAwareResNet


@torch.inference_mode()
def collect_predictions(model, loader, device):
    """Runs a loader once and retains standardized Student-t parameters."""
    model.eval()
    predictions = {
        "hmax": {name: [] for name in ("target", "mu", "sigma", "nu")},
        "arrival": {name: [] for name in ("target", "mu", "sigma", "nu")},
        "slip_id": [],
        "magnitude": [],
    }

    for inputs, targets, metadata in loader:
        inputs = inputs.to(device, non_blocking=True)
        slip_id = metadata["slip_id"].to(device, non_blocking=True)
        output = model(inputs, slip_id)

        for task in ("hmax", "arrival"):
            predictions[task]["target"].append(targets[task].numpy())
            predictions[task]["mu"].append(output[task]["mu"].cpu().numpy())
            predictions[task]["sigma"].append(output[task]["sigma"].cpu().numpy())
            nu = output[task]["nu"].expand_as(output[task]["mu"])
            predictions[task]["nu"].append(nu.cpu().numpy())
        predictions["slip_id"].append(metadata["slip_id"].numpy())
        predictions["magnitude"].append(metadata["mw"].numpy())

    for task in ("hmax", "arrival"):
        for name in predictions[task]:
            predictions[task][name] = np.concatenate(predictions[task][name])
    predictions["slip_id"] = np.concatenate(predictions["slip_id"])
    predictions["magnitude"] = np.concatenate(predictions["magnitude"])
    return predictions


def _load_state_dict(path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    cleaned = {}
    for name, value in state.items():
        if name.startswith("module."):
            name = name.removeprefix("module.")
        if name.startswith("core."):
            name = name.removeprefix("core.")
        cleaned[name] = value
    return cleaned


def predict_checkpoint(
    data_root,
    split_dir,
    run_dir,
    checkpoint=None,
    batch_size=4,
    num_workers=None,
    device=None,
):
    """Loads a checkpoint and returns predictions on the fixed test split."""
    data_root = Path(data_root)
    split_dir = Path(split_dir)
    run_dir = Path(run_dir)
    stats = np.load(run_dir / "preprocessing_stats.npy")
    test_indices = np.load(split_dir / "test_idx.npy")
    dataset = MakranDataset(data_root, test_indices, stats)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 4)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    model = StationAwareResNet(in_channels=len(dataset.channels) + 1).to(device)
    coordinates = station_pixel_coordinates(
        data_root / "additional_files" / "stationsall.npy",
        dataset.bathymetry.shape[-2:],
        stride=dataset.stride,
    )
    model.set_station_coordinates(
        torch.as_tensor(coordinates, dtype=torch.float32, device=device)
    )

    if checkpoint is None:
        checkpoint = run_dir / "best.pt"
        if not checkpoint.exists():
            checkpoint = run_dir / "final.pt"
    model.load_state_dict(_load_state_dict(checkpoint, device), strict=True)
    predictions = collect_predictions(model, loader, device)
    return predictions, stats


def evaluate_checkpoint(
    data_root,
    split_dir,
    run_dir,
    checkpoint=None,
    batch_size=4,
    num_workers=None,
    device=None,
):
    """Evaluates a checkpoint on the fixed test split."""
    predictions, stats = predict_checkpoint(
        data_root,
        split_dir,
        run_dir,
        checkpoint=checkpoint,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    return summarize_predictions(predictions, stats)
