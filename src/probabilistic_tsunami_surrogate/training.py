"""Training utilities for the station-aware surrogate."""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from probabilistic_tsunami_surrogate.data.dataset import (
    MakranDataset,
    station_pixel_coordinates,
)
from probabilistic_tsunami_surrogate.data.preprocessing import fit_preprocessing_stats
from probabilistic_tsunami_surrogate.data.splits import make_micro_validation_split
from probabilistic_tsunami_surrogate.losses import multitask_student_t_nll
from probabilistic_tsunami_surrogate.model import StationAwareResNet


@torch.no_grad()
def initialize_regime_offsets(model):
    """Applies the fixed regime-level location and scale offsets."""
    model.log_scale_regime_h.add_(model.log_scale_regime_h.new_tensor([0.05, 0.15]))
    model.log_scale_regime_t.add_(model.log_scale_regime_t.new_tensor([0.10, 0.06]))
    model.mu_regime_h.add_(model.mu_regime_h.new_tensor([0.04, 0.24]))
    model.mu_regime_t.add_(model.mu_regime_t.new_tensor([-0.05, -0.04]))


def make_optimizer(model, learning_rate=1e-3):
    """Builds the four AdamW parameter groups used for training."""
    scale_h = [model.log_scale_station_h, model.log_scale_regime_h]
    scale_t = [model.log_scale_station_t, model.log_scale_regime_t]
    nu = [model.nu_regime_raw_h, model.nu_regime_raw_t]
    excluded = {id(parameter) for parameter in scale_h + scale_t + nu}
    base = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in excluded
    ]
    return torch.optim.AdamW(
        [
            {"params": base, "lr": learning_rate, "weight_decay": 0.0},
            {"params": scale_h, "lr": learning_rate, "weight_decay": 1e-4},
            {
                "params": scale_t,
                "lr": 0.5 * learning_rate,
                "weight_decay": 1e-4,
            },
            {"params": nu, "lr": 0.2 * learning_rate, "weight_decay": 0.0},
        ]
    )


def _predict_batch(model, batch, device):
    inputs, targets, metadata = batch
    inputs = inputs.to(device, non_blocking=True)
    targets = {
        task: values.to(device, non_blocking=True)
        for task, values in targets.items()
    }
    slip_id = metadata["slip_id"].to(device, non_blocking=True)
    return model(inputs, slip_id), targets


def train_epoch(model, loader, optimizer, device):
    """Trains for one epoch and returns micro-averaged Student-t NLL."""
    model.train()
    loss_sum = 0.0
    value_count = 0

    for batch in loader:
        predictions, targets = _predict_batch(model, batch, device)
        count = sum((~torch.isnan(values)).sum().item() for values in targets.values())
        if count == 0:
            continue

        loss = multitask_student_t_nll(predictions, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_sum += loss.detach().item() * count
        value_count += count

    return loss_sum / value_count if value_count else float("nan")


@torch.inference_mode()
def validation_nll(model, loader, device):
    """Computes per-task and micro-averaged NLL in standardized space."""
    model.eval()
    sums = {
        task: torch.zeros((), dtype=torch.float64, device=device)
        for task in ("hmax", "arrival")
    }
    counts = {
        task: torch.zeros((), dtype=torch.float64, device=device)
        for task in ("hmax", "arrival")
    }

    for batch in loader:
        predictions, targets = _predict_batch(model, batch, device)
        for task in ("hmax", "arrival"):
            target = targets[task]
            observed = ~torch.isnan(target)
            head = predictions[task]
            distribution = torch.distributions.StudentT(
                head["nu"],
                loc=head["mu"],
                scale=head["sigma"],
            )
            sums[task] += (-distribution.log_prob(target)[observed]).sum(
                dtype=torch.float64
            )
            counts[task] += observed.sum()

    output = {
        task: (sums[task] / counts[task]).item()
        for task in ("hmax", "arrival")
    }
    output["micro"] = (
        (sums["hmax"] + sums["arrival"])
        / (counts["hmax"] + counts["arrival"])
    ).item()
    return output


def train_model(
    data_root,
    split_dir,
    run_dir,
    max_epochs=80,
    batch_size=4,
    learning_rate=1e-3,
    micro_validation_size=0.075,
    patience=5,
    soft_kappa=5.0,
    plateau_threshold=4.5e-3,
    plateau_cooldown=1,
    num_workers=None,
    device=None,
):
    """Trains the station-aware model and writes checkpoints and run arrays.

    Model initialization and batch order are intentionally not seeded to retain
    stochastic variation across repeated training runs.
    """
    data_root = Path(data_root)
    split_dir = Path(split_dir)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    split_output = run_dir / "splits"
    split_output.mkdir(exist_ok=True)

    entries = np.load(data_root / "entries.npy", allow_pickle=True)
    train_indices = np.load(split_dir / "train_idx.npy")
    train_core, validation_indices = make_micro_validation_split(
        train_indices,
        entries,
        validation_size=micro_validation_size,
    )
    stats = fit_preprocessing_stats(data_root, train_core, entries=entries)

    train_dataset = MakranDataset(data_root, train_core, stats)
    validation_dataset = MakranDataset(data_root, validation_indices, stats)
    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 4)
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_options,
    )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = StationAwareResNet(in_channels=len(train_dataset.channels) + 1).to(device)
    coordinates = station_pixel_coordinates(
        data_root / "additional_files" / "stationsall.npy",
        train_dataset.bathymetry.shape[-2:],
        stride=train_dataset.stride,
    )
    model.set_station_coordinates(
        torch.as_tensor(coordinates, dtype=torch.float32, device=device)
    )
    initialize_regime_offsets(model)

    optimizer = make_optimizer(model, learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        threshold=plateau_threshold,
        cooldown=plateau_cooldown,
        min_lr=1e-6,
    )

    np.save(split_output / "train_core.npy", train_core)
    np.save(split_output / "micro_validation.npy", validation_indices)
    np.save(run_dir / "preprocessing_stats.npy", np.asarray(stats))

    history = {"train": [], "hmax": [], "arrival": [], "micro": [], "lr": []}
    baseline_samples = {"hmax": [], "arrival": []}
    baselines = {"hmax": None, "arrival": None}
    best_joint = float("inf")
    epochs_without_improvement = 0
    best_path = run_dir / "best.pt"

    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        validation = validation_nll(model, validation_loader, device)
        scheduler.step(validation["micro"])

        history["train"].append(train_loss)
        history["hmax"].append(validation["hmax"])
        history["arrival"].append(validation["arrival"])
        history["micro"].append(validation["micro"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        for task in ("hmax", "arrival"):
            baseline_samples[task].append(validation[task])
            if baselines[task] is None and len(baseline_samples[task]) >= 3:
                baselines[task] = float(np.median(baseline_samples[task][:3]))

        current_baselines = {
            task: baselines[task]
            if baselines[task] is not None
            else float(np.median(baseline_samples[task]))
            for task in ("hmax", "arrival")
        }
        normalized_h = validation["hmax"] / current_baselines["hmax"]
        normalized_t = validation["arrival"] / current_baselines["arrival"]
        maximum = max(normalized_h, normalized_t)
        joint = maximum + np.log(
            0.5
            * (
                np.exp(soft_kappa * (normalized_h - maximum))
                + np.exp(soft_kappa * (normalized_t - maximum))
            )
        ) / soft_kappa

        improved = joint < best_joint * (1 - 5e-3)
        if improved:
            best_joint = joint
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_without_improvement += 1

        print(
            f"epoch={epoch + 1:03d} train={train_loss:.5f} "
            f"validation={validation['micro']:.5f} joint={joint:.5f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        if epochs_without_improvement >= patience:
            break

    final_path = run_dir / "final.pt"
    torch.save(model.state_dict(), final_path)
    np.savez(run_dir / "history.npz", **history)
    return {
        "final_checkpoint": final_path,
        "best_checkpoint": best_path,
        "epochs": len(history["train"]),
    }
