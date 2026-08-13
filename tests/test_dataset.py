import numpy as np
import pytest

torch = pytest.importorskip("torch")

from probabilistic_tsunami_surrogate.data.dataset import (  # noqa: E402
    MakranDataset,
    station_pixel_coordinates,
)


def test_dual_target_dataset(tmp_path):
    scenario_dir = tmp_path / "8.9" / "M1_1"
    (scenario_dir / "external_files").mkdir(parents=True)
    (tmp_path / "additional_files").mkdir()

    np.save(tmp_path / "entries.npy", np.array([("8.9", "M1_1")], dtype=object))
    np.save(tmp_path / "additional_files" / "bathy.npy", np.arange(24).reshape(4, 6))
    np.save(tmp_path / "additional_files" / "stationsall.npy", np.zeros((322, 2)))
    np.save(scenario_dir / "external_files" / "eta.npy", np.ones((4, 6)))
    np.save(scenario_dir / "hmax.npy", np.ones(322))
    np.save(scenario_dir / "arrival_times.npy", np.ones(322))

    dataset = MakranDataset(tmp_path, [0], stats=(0, 1, 0, 1, 0, 1), stride=2)
    inputs, targets, metadata = dataset[0]

    assert inputs.shape == (2, 2, 3)
    assert targets["hmax"].shape == (317,)
    assert targets["arrival"].shape == (317,)
    assert metadata["slip_id"].item() == 1

    coordinates = station_pixel_coordinates(
        tmp_path / "additional_files" / "stationsall.npy",
        grid_shape=inputs.shape[-2:],
        stride=2,
    )
    assert coordinates.shape == (317, 2)
