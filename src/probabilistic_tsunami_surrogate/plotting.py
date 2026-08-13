"""Dataset and evaluation plotting utilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from probabilistic_tsunami_surrogate.data.preprocessing import station_keep_mask
from probabilistic_tsunami_surrogate.metrics import pit_values


def plot_data_distributions(root, entries, output_dir):
    """Writes streaming η and target-distribution plots."""
    root = Path(root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    keep = station_keep_mask()

    eta_bins = np.linspace(-6, 6, 101)
    eta_counts = np.zeros(len(eta_bins) - 1, dtype=np.int64)
    targets = {"hmax": [], "arrival": []}
    for magnitude, scenario_id in entries:
        directory = root / str(magnitude) / str(scenario_id)
        eta = np.load(
            directory / "external_files" / "eta.npy",
            mmap_mode="r",
        )
        finite_eta = np.asarray(eta)[np.isfinite(eta)]
        eta_counts += np.histogram(finite_eta, eta_bins)[0]
        targets["hmax"].append(np.load(directory / "hmax.npy")[keep])
        targets["arrival"].append(
            np.load(directory / "arrival_times.npy")[keep]
        )

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(eta_bins[:-1], eta_counts, width=np.diff(eta_bins), align="edge")
    axis.set(title="Initial surface elevation", xlabel="η (m)", ylabel="Count")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "eta_distribution.png", dpi=160)
    plt.close(figure)

    specifications = {
        "hmax": (np.linspace(-5, 25, 101), np.arcsinh, "arcsinh(hmax)"),
        "arrival": (np.linspace(0, 11_000, 121), np.log1p, "log1p(arrival)"),
    }
    for task, (raw_bins, transform, transformed_label) in specifications.items():
        values = np.concatenate(targets[task])
        values = values[np.isfinite(values)]
        transformed = transform(values)
        lower = np.percentile(transformed, 0.1)
        upper = np.percentile(transformed, 99.9)
        if lower == upper:
            lower, upper = lower - 0.5, upper + 0.5
        transformed_bins = np.linspace(
            lower,
            upper,
            121,
        )
        figure, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(values, bins=raw_bins)
        axes[0].set(title=f"{task} in physical space", xlabel=task, ylabel="Count")
        axes[1].hist(transformed, bins=transformed_bins)
        axes[1].set(
            title=f"Transformed {task}",
            xlabel=transformed_label,
            ylabel="Count",
        )
        for axis in axes:
            axis.grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(output_dir / f"{task}_distribution.png", dpi=160)
        plt.close(figure)


def plot_station_maps(
    bathymetry,
    station_coordinates,
    values,
    output_path,
    title,
    colorbar_label,
):
    """Plots paired station metrics over the bathymetric coastline."""
    x_offset, y_offset, spacing = 55.8, 22.6, 0.006
    height, width = bathymetry.shape
    longitude = x_offset + spacing * np.arange(width)
    latitude = y_offset + spacing * np.arange(height)
    station_longitude = x_offset + spacing * station_coordinates[:, 0]
    station_latitude = y_offset + spacing * station_coordinates[:, 1]

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for axis, (task, metric) in zip(axes, values.items()):
        metric = np.asarray(metric, dtype=float)
        valid = np.isfinite(metric)
        lower, upper = (0.0, 1.0)
        if valid.any():
            lower, upper = np.nanpercentile(metric, [5, 95])
            if lower >= upper:
                lower, upper = np.nanmin(metric), np.nanmax(metric)
            if lower >= upper:
                lower, upper = lower - 0.5, upper + 0.5
        axis.contour(
            longitude,
            latitude,
            bathymetry,
            levels=[0],
            colors="black",
            linewidths=1,
        )
        points = axis.scatter(
            station_longitude,
            station_latitude,
            c=np.ma.masked_invalid(metric),
            cmap="magma",
            s=45,
            edgecolors="black",
            linewidths=0.3,
            vmin=lower,
            vmax=upper,
        )
        axis.set(title=task, xlabel="Longitude", ylabel="Latitude", aspect="equal")
        figure.colorbar(points, ax=axis, label=colorbar_label)
    figure.suptitle(title)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_qq_panels(diagnostics, output_path):
    """Plots PIT-normal Q-Q diagnostics for both tasks."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for axis, (task, (theoretical, observed, stats)) in zip(
        axes,
        diagnostics.items(),
    ):
        lower = min(theoretical.min(), observed.min())
        upper = max(theoretical.max(), observed.max())
        axis.scatter(theoretical, observed, s=4, alpha=0.5)
        axis.plot([lower, upper], [lower, upper], color="black", linewidth=1)
        axis.set(
            title=f"{task} (central slope={stats['central_slope']:.2f})",
            xlabel="Theoretical normal quantile",
            ylabel="PIT-normal quantile",
            aspect="equal",
        )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_coverage_summary(summary, output_path):
    """Plots 50% and 95% coverage with PIT moments."""
    tasks = ("hmax", "arrival")
    x = np.arange(len(tasks))
    width = 0.32
    coverage_50 = [summary[task]["overall"]["picp_50"] for task in tasks]
    coverage_95 = [summary[task]["overall"]["picp_95"] for task in tasks]

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(x - width / 2, coverage_50, width, label="50% interval")
    axis.bar(x + width / 2, coverage_95, width, label="95% interval")
    axis.axhline(0.5, color="grey", linestyle="--", linewidth=1)
    axis.axhline(0.95, color="grey", linestyle="--", linewidth=1)
    axis.set(xticks=x, xticklabels=tasks, ylabel="Empirical coverage", ylim=(0, 1))
    pit_text = " | ".join(
        f"{task}: μ={summary[task]['overall']['pit_mean']:.2f}, "
        f"σ={summary[task]['overall']['pit_std']:.2f}"
        for task in tasks
    )
    axis.set_title(f"PICP and PIT summary\n{pit_text}")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_pit_histograms(predictions, output_path):
    """Plots PIT histograms by task and slip regime."""
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for row, task in enumerate(("hmax", "arrival")):
        for column, slip in enumerate((0, 1)):
            values = pit_values(predictions, task, slip)
            axes[row, column].hist(values, bins=20, range=(0, 1), density=True)
            axes[row, column].axhline(1, color="black", linewidth=1)
            axes[row, column].set(
                title=f"{task}, slip={slip}",
                xlabel="PIT",
                ylabel="Density",
                xlim=(0, 1),
            )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
