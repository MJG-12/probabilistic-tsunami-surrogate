import json

import numpy as np

from probabilistic_tsunami_surrogate.config import RunConfig
from probabilistic_tsunami_surrogate.data.preprocessing import station_keep_mask
from probabilistic_tsunami_surrogate.data.validation import station_target_iqr
from probabilistic_tsunami_surrogate.evaluation import predict_checkpoint
from probabilistic_tsunami_surrogate.metrics import (
    qq_diagnostics,
    station_metrics,
    summarize_predictions,
)
from probabilistic_tsunami_surrogate.plotting import (
    plot_coverage_summary,
    plot_pit_histograms,
    plot_qq_panels,
    plot_station_maps,
)


def build_report_assets(config):
    """Builds numerical and graphical report assets."""
    predictions, stats = predict_checkpoint(
        config.DATA_ROOT,
        config.SPLIT_DIR,
        config.RUN_DIR,
        checkpoint=config.CHECKPOINT,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        device=config.DEVICE,
    )
    config.report_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_predictions(predictions, stats)

    entries = np.load(config.entries_path, allow_pickle=True)
    train_path = config.RUN_DIR / "splits" / "train_core.npy"
    if not train_path.exists():
        train_path = config.SPLIT_DIR / "train_idx.npy"
    target_iqr = station_target_iqr(
        config.DATA_ROOT,
        entries,
        np.load(train_path),
    )
    by_station = station_metrics(predictions, stats, target_iqr)
    np.savez(
        config.report_dir / "station_metrics.npz",
        hmax_mae=by_station["hmax"]["mae"],
        hmax_mae_over_iqr=by_station["hmax"]["mae_over_iqr"],
        hmax_iw95=by_station["hmax"]["iw95_median"],
        arrival_mae=by_station["arrival"]["mae"],
        arrival_mae_over_iqr=by_station["arrival"]["mae_over_iqr"],
        arrival_iw95=by_station["arrival"]["iw95_median"],
    )

    bathymetry = np.load(config.DATA_ROOT / "additional_files" / "bathy.npy")
    coordinates = np.load(
        config.DATA_ROOT / "additional_files" / "stationsall.npy"
    )[station_keep_mask()]
    plot_station_maps(
        bathymetry,
        coordinates,
        {
            "Maximum wave height": by_station["hmax"]["mae_over_iqr"],
            "Arrival time": by_station["arrival"]["mae_over_iqr"],
        },
        config.report_dir / "station_mae_over_iqr.png",
        "Station error normalized by training IQR",
        "MAE / IQR",
    )
    plot_station_maps(
        bathymetry,
        coordinates,
        {
            "Maximum wave height": by_station["hmax"]["iw95_median"],
            "Arrival time": by_station["arrival"]["iw95_median"],
        },
        config.report_dir / "station_iw95.png",
        "Median 95% prediction-interval width",
        "IW95",
    )

    diagnostics = {
        task: qq_diagnostics(predictions, task)
        for task in ("hmax", "arrival")
    }
    plot_qq_panels(diagnostics, config.report_dir / "qq_plots.png")
    plot_coverage_summary(summary, config.report_dir / "coverage_summary.png")
    plot_pit_histograms(predictions, config.report_dir / "pit_by_slip.png")

    output = {
        "metrics": summary,
        "qq": {task: values[2] for task, values in diagnostics.items()},
    }
    with (config.report_dir / "summary.json").open("w") as stream:
        json.dump(output, stream, indent=2)
    print(f"report assets: {config.report_dir}")


if __name__ == "__main__":
    build_report_assets(RunConfig())
