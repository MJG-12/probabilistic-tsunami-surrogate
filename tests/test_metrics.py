import sys
import types

import numpy as np

from probabilistic_tsunami_surrogate.metrics import (
    inverse_transform,
    qq_diagnostics,
    station_metrics,
    summarize_predictions,
)


def test_inverse_target_transforms():
    values = np.array([-1.0, 0.0, 1.0])
    np.testing.assert_allclose(
        inverse_transform(values, 0, 1, "hmax"),
        np.sinh(values),
    )
    np.testing.assert_allclose(
        inverse_transform(values, 0, 1, "arrival"),
        np.expm1(values),
    )


def test_report_summary_groups_predictions(monkeypatch):
    scoringrules = types.ModuleType("scoringrules")
    scoringrules.crps_t = lambda target, nu, location, scale: np.abs(
        target - location
    )
    monkeypatch.setitem(sys.modules, "scoringrules", scoringrules)

    task = {
        "target": np.zeros((2, 2)),
        "mu": np.array([[0.0, 0.1], [0.2, 0.3]]),
        "sigma": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "nu": np.full((2, 2), 4.0),
    }
    predictions = {
        "hmax": task,
        "arrival": {name: values.copy() for name, values in task.items()},
        "slip_id": np.array([0, 1]),
        "magnitude": np.array([8.8, 9.0]),
    }

    summary = summarize_predictions(predictions, stats=(0, 1, 0, 1, 0, 1))

    assert summary["hmax"]["overall"]["n"] == 4
    assert summary["arrival"]["by_slip"]["0"]["n"] == 2
    assert summary["arrival"]["overall"]["picp_95"] == 1

    by_station = station_metrics(
        predictions,
        stats=(0, 1, 0, 1, 0, 1),
        target_iqr={"hmax": np.ones(2), "arrival": np.ones(2)},
    )
    assert by_station["hmax"]["mae"].shape == (2,)
    assert by_station["arrival"]["iw95_median"].shape == (2,)

    theoretical, observed, diagnostics = qq_diagnostics(predictions, "hmax")
    assert theoretical.shape == observed.shape == (4,)
    assert diagnostics["n"] == 4
