from dataclasses import replace

import numpy as np

from probabilistic_tsunami_surrogate.config import RunConfig
from probabilistic_tsunami_surrogate.data.cleaning import (
    extract_station_targets,
    fix_number_format,
)
from probabilistic_tsunami_surrogate.data.preprocessing import (
    fit_preprocessing_stats,
    load_elevation_vectors,
    safe_log1p,
    slip_id_from_sid,
    station_keep_mask,
)
from probabilistic_tsunami_surrogate.data.splits import (
    make_micro_validation_split,
    make_train_test_split,
)


def test_fixed_station_and_slip_conventions():
    assert station_keep_mask().sum() == 317
    assert slip_id_from_sid("8.8", "M1") == 0
    assert slip_id_from_sid("8.9", "M1") == 0
    assert slip_id_from_sid("8.9", "M1_1_200") == 1
    assert slip_id_from_sid("9.0", "M1") == 1


def test_run_config_derives_local_paths(tmp_path):
    config = replace(RunConfig(), DATA_ROOT=tmp_path, RUN_DIR=tmp_path / "run")
    assert config.entries_path == tmp_path / "entries.npy"
    assert config.evaluation_path == tmp_path / "run" / "evaluation.json"


def test_target_cleaning_and_arrival_time():
    values = safe_log1p(np.array([0.0, 1.0, -1.0, np.inf]))
    assert np.isfinite(values[:2]).all()
    assert np.isnan(values[2:]).all()
    assert fix_number_format("-0.16792338-117") == "-0.16792338E-117"

    time = np.arange(0.0, 50.0, 10.0)
    eta = np.array([0.001, 0.003, 0.004, 0.010, 0.020])
    data = np.column_stack([time, eta, np.zeros(5), np.zeros(5)])
    hmax, arrival, adjusted = extract_station_targets(data)

    assert np.isclose(hmax, 0.020)
    assert arrival == 30.0
    assert adjusted == 30.0


def test_invalid_elevation_grid_is_skipped(tmp_path):
    valid_path = tmp_path / "8.0" / "M1" / "external_files"
    valid_path.mkdir(parents=True)
    np.save(valid_path / "eta.npy", np.arange(16).reshape(4, 4))

    entries, matrix, issues = load_elevation_vectors(
        tmp_path,
        [("8.0", "M1"), ("8.0", "M2")],
        stride=2,
    )

    assert entries == [("8.0", "M1")]
    assert matrix.shape == (1, 4)
    assert issues[0][:2] == ("8.0", "M2")


def test_split_is_deterministic_and_disjoint():
    entries = np.array([
        (str(magnitude), f"M{index}")
        for magnitude in (8.0, 8.5, 9.0)
        for index in range(40)
    ])
    train, test = make_train_test_split(entries)
    train_again, test_again = make_train_test_split(entries)

    np.testing.assert_array_equal(train, train_again)
    np.testing.assert_array_equal(test, test_again)
    assert set(train).isdisjoint(test)

    core, validation = make_micro_validation_split(train, entries)
    assert len(validation) == round(len(train) * 0.075)
    assert set(core).isdisjoint(validation)
    assert set(core) | set(validation) == set(train)


def test_preprocessing_stats_use_selected_scenarios(tmp_path):
    entries = np.array(
        [("8.8", "M1"), ("8.9", "M2"), ("9.0", "M3")],
        dtype=object,
    )
    np.save(tmp_path / "entries.npy", entries)

    for (magnitude, scenario_id), eta, hmax, arrival in zip(
        entries,
        (0.0, 2.0, 100.0),
        (1.0, 3.0, 100.0),
        (4.0, 8.0, 100.0),
    ):
        scenario_dir = tmp_path / magnitude / scenario_id
        (scenario_dir / "external_files").mkdir(parents=True)
        np.save(scenario_dir / "external_files" / "eta.npy", np.full((2, 2), eta))
        np.save(scenario_dir / "hmax.npy", np.full(322, hmax))
        np.save(scenario_dir / "arrival_times.npy", np.full(322, arrival))

    stats = fit_preprocessing_stats(
        tmp_path,
        indices=[0, 1],
        eta_sample_ratio=1.0,
    )

    expected_hmax = np.arcsinh([1.0, 3.0])
    expected_arrival = np.log1p([4.0, 8.0])
    np.testing.assert_allclose(stats[:2], [1.0, 2.0])
    np.testing.assert_allclose(stats[2:4], [expected_hmax.mean(), expected_hmax.std()])
    np.testing.assert_allclose(
        stats[4:],
        [expected_arrival.mean(), expected_arrival.std()],
    )
