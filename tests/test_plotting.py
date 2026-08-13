import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from probabilistic_tsunami_surrogate.plotting import (  # noqa: E402
    plot_coverage_summary,
    plot_pit_histograms,
    plot_qq_panels,
    plot_station_maps,
)


def test_report_plots_are_written(tmp_path):
    station_values = {
        "hmax": np.array([1.0, 1.0, np.nan]),
        "arrival": np.array([2.0, 3.0, 4.0]),
    }
    plot_station_maps(
        np.linspace(-1, 1, 120).reshape(10, 12),
        np.array([[1, 1], [5, 4], [10, 8]]),
        station_values,
        tmp_path / "stations.png",
        "Station metric",
        "Value",
    )

    diagnostics = {
        task: (
            np.array([-1.0, 0.0, 1.0]),
            np.array([-0.8, 0.1, 1.2]),
            {"central_slope": 1.0},
        )
        for task in ("hmax", "arrival")
    }
    plot_qq_panels(diagnostics, tmp_path / "qq.png")

    summary = {
        task: {
            "overall": {
                "picp_50": 0.5,
                "picp_95": 0.95,
                "pit_mean": 0.5,
                "pit_std": 0.29,
            }
        }
        for task in ("hmax", "arrival")
    }
    plot_coverage_summary(summary, tmp_path / "coverage.png")

    task = {
        "target": np.zeros((2, 3)),
        "mu": np.zeros((2, 3)),
        "sigma": np.ones((2, 3)),
        "nu": np.full((2, 3), 4.0),
    }
    predictions = {
        "hmax": task,
        "arrival": task,
        "slip_id": np.array([0, 1]),
    }
    plot_pit_histograms(predictions, tmp_path / "pit.png")

    assert {path.name for path in tmp_path.iterdir()} == {
        "coverage.png",
        "pit.png",
        "qq.png",
        "stations.png",
    }
