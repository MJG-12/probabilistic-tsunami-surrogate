import numpy as np

from probabilistic_tsunami_surrogate.data.validation import (
    station_target_iqr,
    validate_dataset,
    validate_scenario,
)


def write_scenario(root, magnitude, scenario, eta_value, target_value):
    directory = root / magnitude / scenario
    (directory / "external_files").mkdir(parents=True)
    np.save(directory / "external_files" / "eta.npy", np.full((2, 3), eta_value))
    target = np.full(322, target_value, dtype=np.float32)
    target[np.array([28, 83, 152, 81, 48]) - 1] = np.nan
    np.save(directory / "hmax.npy", target)
    np.save(directory / "arrival_times.npy", target)


def test_scenario_validation_ignores_expected_empty_stations(tmp_path):
    write_scenario(tmp_path, "8.8", "M1", eta_value=1, target_value=2)

    issues = validate_scenario(tmp_path, ("8.8", "M1"), grid_shape=(2, 3))
    valid, dataset_issues = validate_dataset(
        tmp_path,
        [("8.8", "M1")],
        grid_shape=(2, 3),
    )

    assert issues == []
    assert valid == [("8.8", "M1")]
    assert dataset_issues == []


def test_station_iqr_uses_training_entries(tmp_path):
    write_scenario(tmp_path, "8.8", "M1", eta_value=1, target_value=1)
    write_scenario(tmp_path, "8.9", "M2", eta_value=2, target_value=3)
    entries = np.array([("8.8", "M1"), ("8.9", "M2")], dtype=object)

    iqr = station_target_iqr(tmp_path, entries, min_valid=1)

    np.testing.assert_allclose(iqr["hmax"], 1)
    np.testing.assert_allclose(iqr["arrival"], 1)
